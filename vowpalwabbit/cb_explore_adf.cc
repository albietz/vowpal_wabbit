#include "reductions.h"
#include "cb_adf.h"
#include "rand48.h"
#include "bs.h"
#include "gen_cs_example.h"
#include "cb_explore.h"

using namespace LEARNER;
using namespace ACTION_SCORE;
using namespace std;
using namespace CB_ALGS;
//All exploration algorithms return a vector of id, probability tuples, sorted in order of scores. The probabilities are the probability with which each action should be replaced to the top of the list.

//tau first
#define EXPLORE_FIRST 0
//epsilon greedy
#define EPS_GREEDY 1
// bagging explorer
#define BAG_EXPLORE 2
//softmax
#define SOFTMAX 3
//cover
#define COVER 4
//agree
#define AGREE 5
//regcb
#define REGCB 6

#define B_SEARCH_MAX_ITER 20

namespace CB_EXPLORE_ADF
{

struct cb_explore_adf
{
  v_array<example*> ec_seq;
  v_array<action_score> action_probs;

  size_t explore_type;

  size_t tau;
  float epsilon;
  size_t bag_size;
  size_t cover_size;
  float psi;
  bool nounif;
  bool nounifagree;
  float lambda;
  uint64_t offset;
  bool greedify;
  bool randomtie;
  float agree_c0;
  bool regcbopt; // use optimistic variant of RegCB
  float c0;

  size_t counter;

  bool need_to_clear;
  vw* all;
  LEARNER::base_learner* cs_ldf_learner;

  GEN_CS::cb_to_cs_adf gen_cs;
  COST_SENSITIVE::label cs_labels;
  v_array<CB::label> cb_labels;

  CB::label action_label;
  CB::label empty_label;

  COST_SENSITIVE::label cs_labels_2;

  v_array<COST_SENSITIVE::label> prepped_cs_labels;

  // for disagreement
  std::vector<float> disagree_weights;
  std::vector<bool> explored_actions;

  // for random tie breaking
  std::set<uint32_t> tied_actions;

  // for RegCB
  std::vector<float> min_costs;
  std::vector<float> max_costs;

  // for backing up cb example data when computing sensitivities
  std::vector<ACTION_SCORE::action_scores> ex_as;
  std::vector<v_array<CB::cb_class>> ex_costs;
};

template<class T> void swap(T& ele1, T& ele2)
{
  T temp = ele2;
  ele2 = ele1;
  ele1 = temp;
}
template<bool is_learn>
void multiline_learn_or_predict(base_learner& base, v_array<example*>& examples, uint64_t offset, uint32_t id = 0)
{
  for (example* ec : examples)
  {
    uint64_t old_offset = ec->ft_offset;
    ec->ft_offset = offset;
    if (is_learn)
      base.learn(*ec, id);
    else
      base.predict(*ec, id);
    ec->ft_offset = old_offset;
  }
}

example* test_adf_sequence(v_array<example*>& ec_seq)
{
  uint32_t count = 0;
  example* ret = nullptr;
  for (size_t k = 0; k < ec_seq.size(); k++)
  {
    example *ec = ec_seq[k];

    if (ec->l.cb.costs.size() > 1)
      THROW("cb_adf: badly formatted example, only one cost can be known.");

    if (ec->l.cb.costs.size() == 1 && ec->l.cb.costs[0].cost != FLT_MAX)
    {
      ret = ec;
      count += 1;
    }

    if (CB::ec_is_example_header(*ec))
      if (k != 0)
        THROW("warning: example headers at position " << k << ": can only have in initial position!");
  }
  if (count == 0 || count == 1)
    return ret;
  else
    THROW("cb_adf: badly formatted example, only one line can have a cost");
}

void get_disagree_weights_cs(std::vector<float>& weights, cb_explore_adf& data, base_learner& base, v_array<example*>& examples)
{
  const bool shared = CB::ec_is_example_header(*examples[0]);
  const size_t num_actions = examples[0]->pred.a_s.size();
  // for each action, sensitivity with label=min and max
  static std::vector<pair<float, float>> sensitivities;

  auto& ex_as = data.ex_as;
  auto& ex_costs = data.ex_costs;
  ex_as.clear();
  ex_costs.clear();
  sensitivities.resize(num_actions);

  for (auto ex : examples)
  {
    ex_as.push_back(ex->pred.a_s);
    ex_costs.push_back(ex->l.cb.costs);
  }
  const float min_score = data.all->sd->min_cb_cost;
  const float max_score = data.all->sd->max_cb_cost;
  for (auto as : ex_as[0])
  {
    // set regressor prediction
    // examples[shared + as.action]->pred.scalar = as.score;
    examples[shared + as.action]->pred.scalar = min(max_score, max(min_score, as.score));
  }
  // cout << min_score << " " << max_score << endl;

  // compute sensitivities
  for (size_t a = 0; a < num_actions; ++a)
  {
    example* ec = examples[shared + a];
    ec->l.simple.label = min_score;
    if (ec->l.simple.label == ec->pred.scalar)
      sensitivities[a].first = 0;
    else
      sensitivities[a].first = base.sensitivity(*ec);
    // cout << "min: " << ec->pred.scalar << " -> " << ec->l.simple.label << " "
    //      << sensitivities[a].first << ", ";
    ec->l.simple.label = max_score;
    if (ec->l.simple.label == ec->pred.scalar)
      sensitivities[a].second = 0;
    else
      sensitivities[a].second = base.sensitivity(*ec);
    // cout << "max: " << ec->pred.scalar << " -> " << ec->l.simple.label << " "
    //      << sensitivities[a].second << ", ";
  }

  // compute importance weights
  weights.resize(num_actions);
  for (size_t a_ref = 0; a_ref < num_actions; ++a_ref)
  {
    float max_weight = -FLT_MAX;
    float s_ref = sensitivities[a_ref].first;
    float y = examples[shared + a_ref]->pred.scalar;
    for (size_t a = 0; a < num_actions; ++a)
    {
      if (a == a_ref) continue;
      float s = sensitivities[a].second;
      float w;
      if (y <= examples[shared + a]->pred.scalar)
        w = 0;
      else if (s + s_ref == 0)
        w = FLT_MAX;
      else
        w = (y - examples[shared + a]->pred.scalar) / (s_ref + s);
      if (w > max_weight) max_weight = w;
    }
    // cout << max_weight << ", ";
    weights[a_ref] = max_weight * (max_score - min_score) / data.counter;
  }
  // cout << endl;

  // reset examples
  for (size_t i = 0; i < examples.size(); ++i)
  {
    examples[i]->pred.a_s = ex_as[i];
    examples[i]->l.cb.costs = ex_costs[i];
  }
}

void get_disagree_weights_mtr(std::vector<float>& weights, cb_explore_adf& data, base_learner& base, v_array<example*>& examples)
{
  const bool shared = CB::ec_is_example_header(*examples[0]);
  const size_t num_actions = examples[0]->pred.a_s.size();
  auto& ex_as = data.ex_as;
  auto& ex_costs = data.ex_costs;
  ex_as.clear();
  ex_costs.clear();

  for (auto& ex : examples)
  { ex_as.push_back(ex->pred.a_s);
    ex_costs.push_back(ex->l.cb.costs);
  }
  const float min_score = data.all->sd->min_cb_cost;
  const float max_score = data.all->sd->max_cb_cost;
  for (auto as : ex_as[0])
  {
    // set regressor prediction
    examples[shared + as.action]->pred.scalar = as.score;
  }

  // compute importance weights
  weights.resize(num_actions);
  for (size_t a_ref = 0; a_ref < num_actions; ++a_ref)
  { example* ec = examples[shared + a_ref];
    ec->l.simple.label = min_score;
    weights[a_ref] = (examples[shared + a_ref]->pred.scalar - min_score) / base.sensitivity(*ec);
    weights[a_ref] *= (max_score - min_score) / data.counter;
  }

  // reset examples
  for (size_t i = 0; i < examples.size(); ++i)
  { examples[i]->pred.a_s = ex_as[i];
    examples[i]->l.cb.costs = ex_costs[i];
  }
}

void get_disagree_weights(std::vector<float>& weights, cb_explore_adf& data, base_learner& base, v_array<example*>& examples)
{ if (data.gen_cs.cb_type == CB_TYPE_MTR)
    get_disagree_weights_mtr(weights, data, base, examples);
  else
    get_disagree_weights_cs(weights, data, base, examples);
}

// TODO: same as cs_active.cc, move to shared place
float binary_search(float fhat, float delta, float sens, float tol=1e-6)
{
  const float maxw = min(fabs(fhat) / sens, FLT_MAX);

  if (maxw * fhat * fhat <= delta)
    return maxw;

  float l = 0;
  float u = maxw;
  float w, v;

  for (int iter = 0; iter < B_SEARCH_MAX_ITER; iter++)
  {
    w = (u + l) / 2.f;
    v = w * (fhat * fhat - (fhat - sens * w) * (fhat - sens * w)) - delta;
    if (v > 0)
      u = w;
    else
      l = w;
    if (fabs(v) <= tol || u - l <= tol)
      break;
  }

  return l;
}

void get_cost_ranges(std::vector<float> &min_costs,
                     std::vector<float> &max_costs, float delta,
                     cb_explore_adf &data, base_learner &base,
                     v_array<example*>& examples, bool min_only)
{
  const bool shared = CB::ec_is_example_header(*examples[0]);
  const size_t num_actions = examples[0]->pred.a_s.size();
  min_costs.resize(num_actions);
  max_costs.resize(num_actions);

  auto& ex_as = data.ex_as;
  auto& ex_costs = data.ex_costs;
  ex_as.clear();
  ex_costs.clear();

  // backup cb example data
  for (auto& ex : examples)
  {
    ex_as.push_back(ex->pred.a_s);
    ex_costs.push_back(ex->l.cb.costs);
  }

  // set regressor predictions
  for (auto as : ex_as[0])
  {
    examples[shared + as.action]->pred.scalar = as.score;
  }

  const float cmin = data.all->sd->min_cb_cost;
  const float cmax = data.all->sd->max_cb_cost;

  for (size_t a = 0; a < num_actions; ++a)
  {
    example* ec = examples[shared + a];
    ec->l.simple.label = cmin - 1;
    float sens = base.sensitivity(*ec);
    // cout << sens << endl;
    float w = 0; // importance weight

    if (ec->pred.scalar < cmin || nanpattern(sens) || infpattern(sens))
      min_costs[a] = cmin;
    else
    {
      w = binary_search(ec->pred.scalar - cmin - 1, delta, sens);
      min_costs[a] = max(ec->pred.scalar - sens * w, cmin);
      if (min_costs[a] > cmax)
        min_costs[a] = cmax;
    }
      cout << ec->pred.scalar << " " << sens << " " << w << " " << min_costs[a]
           << " | ";

    if (!min_only)
    {
      ec->l.simple.label = cmax + 1;
      sens = base.sensitivity(*ec);
    // cout << sens << endl;
      if (ec->pred.scalar > cmax || nanpattern(sens) || infpattern(sens))
      {
        max_costs[a] = cmax;
      }
      else
      {
        w = binary_search(cmax + 1 - ec->pred.scalar, delta, sens);
        max_costs[a] = min(ec->pred.scalar + sens * w, cmax);
        if (max_costs[a] < cmin)
          max_costs[a] = cmin;
      }
      // cout << sens << " " << w << " " << max_costs[a] << ", ";
    }
  }
  cout << endl;

  // reset cb example data
  for (size_t i = 0; i < examples.size(); ++i)
  { examples[i]->pred.a_s = ex_as[i];
    examples[i]->l.cb.costs = ex_costs[i];
  }
}

void fill_tied(cb_explore_adf& data, v_array<action_score>& preds)
{
  if (!data.randomtie)
    return;

  data.tied_actions.clear();
  for (size_t i = 0; i < preds.size(); ++i)
    if (i == 0 || preds[i].score == preds[0].score)
      data.tied_actions.insert(preds[i].action);
  // cout << "tied: " << data.tied_actions.size() << endl;
}

template <bool is_learn>
void predict_or_learn_first(cb_explore_adf& data, base_learner& base, v_array<example*>& examples)
{
  //Explore tau times, then act according to optimal.
  if (is_learn && data.gen_cs.known_cost.probability < 1 && test_adf_sequence(data.ec_seq) != nullptr)
    multiline_learn_or_predict<true>(base, examples, data.offset);
  else
    multiline_learn_or_predict<false>(base, examples, data.offset);

  v_array<action_score>& preds = examples[0]->pred.a_s;
  uint32_t num_actions = (uint32_t)preds.size();

  if (data.tau)
  {
    float prob = 1.f / (float)num_actions;
    for (size_t i = 0; i < num_actions; i++)
      preds[i].score = prob;
    data.tau--;
  }
  else
  {
    for (size_t i = 1; i < num_actions; i++)
      preds[i].score = 0.;
    preds[0].score = 1.0;
  }
  CB_EXPLORE::safety(preds, data.epsilon, true);
}

template <bool is_learn>
void predict_or_learn_greedy(cb_explore_adf& data, base_learner& base, v_array<example*>& examples)
{
  //Explore uniform random an epsilon fraction of the time.
  std::vector<bool>& explored_actions = data.explored_actions;
  if (is_learn && test_adf_sequence(data.ec_seq) != nullptr)
  {
    // for active variant, impute loss = 1 for unexplored actions (not for MTR)
    if (data.nounifagree && data.gen_cs.cb_type != CB_TYPE_MTR
        && explored_actions.size() == examples[0]->pred.a_s.size())
    {
      uint32_t shared = static_cast<uint32_t>(CB::ec_is_example_header(*examples[0]));
      for (size_t i = shared; i < examples.size() - 1; ++i)
      {
        // cout << i - shared << ":" << examples[i]->l.cb.costs.size()
        //      << ":" << !explored_actions[i - shared] << " ";
        if (!explored_actions[i - shared])
        {
          CB::label& ld = examples[i]->l.cb;
          // cout << i - shared << " ";
          if (ld.costs.size() > 0)
          {
            cout << "cost: " << ld.costs[0].cost << endl;
            THROW("unexplored action has cb label!");
          }
          CB::cb_class cl;
          cl.action = 1 + i - shared;
          // cl.cost = 1.f;
          cl.cost = data.all->sd->max_cb_cost;
          cl.probability = 0;
          ld.costs.push_back(cl);
        }
      }
      // cout << "(imputed losses)" << &explored_actions << endl;
    }
    if (data.nounifagree)
      data.all->nounifagree = true;
    multiline_learn_or_predict<true>(base, examples, data.offset);
    if (data.nounifagree)
      data.all->nounifagree = false;
  }
  else
    multiline_learn_or_predict<false>(base, examples, data.offset);
  
  v_array<action_score>& preds = examples[0]->pred.a_s;
  uint32_t num_actions = (uint32_t)preds.size();
  ++data.counter;
  if (data.randomtie)
  {
    fill_tied(data, preds);
  }

  if (data.nounifagree && !is_learn)
  {
    std::vector<float>& weights = data.disagree_weights;
    get_disagree_weights(weights, data, base, examples);

    size_t support_size = 0;
    // const float threshold = data.agree_c0 / (float)sqrt(data.counter * num_actions);
    if (data.epsilon == 0)
      THROW("need non-zero epsilon for disagreement test");
    const float et = data.agree_c0 * num_actions * log(data.counter) / (data.epsilon * data.counter);
    const float threshold = sqrt(et) + et;
    // cout << threshold << " - ";
    /*for (size_t i = 0; i < num_actions; ++i)
      std::cout << weights[i] << "(" << (weights[i] <= threshold) << ") ";
    std::cout << std::endl;*/
    explored_actions.resize(num_actions);
    for (size_t i = 0; i < num_actions; ++i)
    {
      if (weights[preds[i].action] > threshold)
      {
        preds[i].score = 0;
        if (i == 0 ||
            (data.randomtie &&
             data.tied_actions.count(preds[i].action) > 0)) // greedy action(s)
        {
          explored_actions[preds[i].action] = true;
        }
        else
        {
          explored_actions[preds[i].action] = false;
          // cout << preds[i].action << " ";
        }
      }
      else
      {
        // std::cout << preds[i].action << " ";
        preds[i].score = 1;
        explored_actions[preds[i].action] = true;
        ++support_size;
      }
    }
    // cout << endl;
    // cout << "(disagreeing actions)" << &explored_actions << endl;

    // exploration mass
    float prob = data.epsilon / num_actions;
    float expl_mass = 0.f;
    for (size_t i = 0; i < num_actions; ++i)
    {
      if (preds[i].score > 0)
      {
        preds[i].score = prob;
        expl_mass += prob;
      }
    }

    // greedy mass
    if (data.randomtie)
    {
      for (size_t i = 0; i < num_actions; ++i)
        if (data.tied_actions.count(preds[i].action) > 0)
          preds[i].score += (1.f - expl_mass) / data.tied_actions.size();
    }
    else
      preds[0].score += 1 - expl_mass;
  }
  else
  {
    float prob = data.epsilon/(float)num_actions;
    for (size_t i = 0; i < num_actions; i++)
      preds[i].score = prob;
    if (data.randomtie)
    {
      for (size_t i = 0; i < num_actions; ++i)
        if (data.tied_actions.count(preds[i].action) > 0)
          preds[i].score += (1.f - data.epsilon) / data.tied_actions.size();
    }
    else
      preds[0].score += 1.f - data.epsilon;
  }
}

template <bool is_learn>
void predict_or_learn_regcb(cb_explore_adf& data, base_learner& base, v_array<example*>& examples)
{
  if (is_learn && test_adf_sequence(examples) != nullptr)
  {
    uint32_t shared = static_cast<uint32_t>(CB::ec_is_example_header(*examples[0]));
    for (size_t i = shared; i < examples.size() - 1; ++i)
    {
      CB::label& ld = examples[i]->l.cb;
      if (ld.costs.size() == 1)
      {
        ld.costs[0].probability = 1.f; // no importance weighting
      }
    }

    multiline_learn_or_predict<true>(base, examples, data.offset);
  }
  else
    multiline_learn_or_predict<false>(base, examples, data.offset);

  v_array<action_score>& preds = examples[0]->pred.a_s;
  uint32_t num_actions = (uint32_t)preds.size();
  ++data.counter;

  const float max_range = data.all->sd->max_cb_cost - data.all->sd->min_cb_cost;
  // threshold on empirical loss difference
  const float delta =
      data.c0 * log((float)(num_actions * data.counter)) * pow(max_range, 2);

  if (!is_learn)
  {
    get_cost_ranges(data.min_costs, data.max_costs, delta, data, base, examples,
                    /*min_only=*/data.regcbopt);

    for (size_t i = 0; i < num_actions; ++i)
    {
      cout << "(" << data.min_costs[preds[i].action] << ", "
        << preds[i].score << ", " << (data.regcbopt ? 0. : data.max_costs[preds[i].action])
        << ") ";
    }
    cout << endl;

    if (data.regcbopt) // optimistic variant
    {
      float min_cost = FLT_MAX;
      size_t a_opt = 0;  // optimistic action
      for (size_t a = 0; a < num_actions; ++a)
      {
        if (data.min_costs[a] < min_cost)
        {
          min_cost = data.min_costs[a];
          a_opt = a;
        }
      }
      for (size_t i = 0; i < preds.size(); ++i)
      {
        if (preds[i].action == a_opt ||
            (data.randomtie && data.min_costs[preds[i].action] == min_cost))
          preds[i].score = 1;
        else
          preds[i].score = 0;
      }
      // explore uniformly on support (random tie breaking)
      CB_EXPLORE::safety(preds, 1.0, /*zeros=*/false);
    }
    else // elimination variant
    {
      float min_max_cost = FLT_MAX;
      for (size_t a = 0; a < num_actions; ++a)
        if (data.max_costs[a] < min_max_cost)
          min_max_cost = data.max_costs[a];
      // cout << preds.size() << " min max: " << min_max_cost << endl;
      for (size_t i = 0; i < preds.size(); ++i)
      {
        if (data.min_costs[preds[i].action] <= min_max_cost)
          preds[i].score = 1;
        else
          preds[i].score = 0;
        // explore uniformly on support
        CB_EXPLORE::safety(preds, 1.0, /*zeros=*/false);
      }
    }
  }
}

template <bool is_learn>
void predict_or_learn_bag(cb_explore_adf& data, base_learner& base, v_array<example*>& examples)
{
  //Randomize over predictions from a base set of predictors
  v_array<action_score>& preds = examples[0]->pred.a_s;
  uint32_t num_actions = (uint32_t)(examples.size() - 1);
  if (CB::ec_is_example_header(*examples[0]))
    num_actions--;
  if (num_actions == 0)
  {
    preds.erase();
    return;
  }

  data.action_probs.resize(num_actions);
  data.action_probs.erase();
  for (uint32_t i = 0; i < num_actions; i++)
    data.action_probs.push_back({ i,0. });
  float prob = 1.f / (float)data.bag_size;
  bool test_sequence = test_adf_sequence(data.ec_seq) == nullptr;
  for (uint32_t i = 0; i < data.bag_size; i++)
  {
    const uint32_t id = (i == 0) ? i : i + 1; // skip DR policy
    // avoid updates to the random num generator
    // for greedify, always update first policy once
    uint32_t count = is_learn
                     ? ((data.greedify && i == 0) ? 1 : BS::weight_gen(*data.all))
                     : 0;
    if (is_learn && count > 0 && !test_sequence)
      multiline_learn_or_predict<true>(base, examples, data.offset, id);
    else
      multiline_learn_or_predict<false>(base, examples, data.offset, id);
    assert(preds.size() == num_actions);
    if (data.randomtie)
    {
      fill_tied(data, preds);
      for (uint32_t a : data.tied_actions)
        data.action_probs[a].score += prob / data.tied_actions.size();
    }
    else
      data.action_probs[preds[0].action].score += prob;
    if (is_learn && !test_sequence)
      for (uint32_t j = 1; j < count; j++)
        multiline_learn_or_predict<true>(base, examples, data.offset, id);
  }

  CB_EXPLORE::safety(data.action_probs, data.epsilon, true);
  qsort((void*) data.action_probs.begin(), data.action_probs.size(), sizeof(action_score), reverse_order);

  for (size_t i = 0; i < num_actions; i++)
    preds[i] = data.action_probs[i];
}

template <bool is_learn>
void predict_or_learn_cover(cb_explore_adf& data, base_learner& base, v_array<example*>& examples)
{
  //Randomize over predictions from a base set of predictors
  //Use cost sensitive oracle to cover actions to form distribution.
  const bool is_mtr = data.gen_cs.cb_type == CB_TYPE_MTR;
  if (is_learn)
  {
    if (is_mtr) // use DR estimates for non-ERM policies in MTR
      GEN_CS::gen_cs_example_dr<true>(data.gen_cs, examples, data.cs_labels);
    else
      GEN_CS::gen_cs_example<false>(data.gen_cs, examples, data.cs_labels);
    multiline_learn_or_predict<true>(base, examples, data.offset);
  }
  else
  {
    GEN_CS::gen_cs_example_ips(examples, data.cs_labels);
    multiline_learn_or_predict<false>(base, examples, data.offset);
  }

  static std::vector<float> weights;
  if (data.nounifagree)
  { get_disagree_weights(weights, data, base, examples);
  }

  v_array<action_score>& preds = examples[0]->pred.a_s;
  const uint32_t num_actions = (uint32_t)preds.size();

  float additive_probability = 1.f / (float)data.cover_size;
  const float min_prob = min(1.f / num_actions, 1.f / (float)sqrt(data.counter * num_actions));
  v_array<action_score>& probs = data.action_probs;
  probs.erase();
  for(uint32_t i = 0; i < num_actions; i++)
    probs.push_back({i,0.});

  if (false && data.randomtie)
  {
    fill_tied(data, preds);
    for (uint32_t a : data.tied_actions)
      probs[a].score += additive_probability / data.tied_actions.size();
  }
  else
    probs[preds[0].action].score += additive_probability;

  const uint32_t shared = CB::ec_is_example_header(*examples[0]) ? 1 : 0;

  float norm = min_prob * num_actions + (additive_probability - min_prob);
  for (size_t i = 1; i < data.cover_size; i++)
  {
    //Create costs of each action based on online cover
    if (is_learn)
    {
      data.cs_labels_2.costs.erase();
      if (shared > 0)
        data.cs_labels_2.costs.push_back(data.cs_labels.costs[0]);
      for (uint32_t j = 0; j < num_actions; j++)
      {
        float pseudo_cost = data.cs_labels.costs[j+shared].x - data.psi * min_prob / (max(probs[j].score, min_prob) / norm);
        data.cs_labels_2.costs.push_back({pseudo_cost,j,0.,0.});
      }
      GEN_CS::call_cs_ldf<true>(*(data.cs_ldf_learner), examples, data.cb_labels, data.cs_labels_2, data.prepped_cs_labels, data.offset, i+1);
    }
    else
      GEN_CS::call_cs_ldf<false>(*(data.cs_ldf_learner), examples, data.cb_labels, data.cs_labels, data.prepped_cs_labels, data.offset, i+1);

    if (false && data.randomtie)
    {
      fill_tied(data, preds);
      const float add_prob = additive_probability / data.tied_actions.size();
      for (uint32_t a : data.tied_actions)
      {
        if (probs[a].score < min_prob)
          norm += max(0, add_prob - (min_prob - probs[a].score));
        else
          norm += add_prob;
        probs[a].score += add_prob;
      }
    }
    else
    {
      uint32_t action = preds[0].action;
      if (probs[action].score < min_prob)
        norm += max(0, additive_probability - (min_prob - probs[action].score));
      else
        norm += additive_probability;
      probs[action].score += additive_probability;
    }
  }

  if (data.nounifagree)
  { const float threshold = data.agree_c0 / (float)sqrt((data.counter + 1) * num_actions);
    for (size_t i = 0; i < num_actions; ++i)
    { if (probs[i].score == 0 && weights[probs[i].action] <= threshold)
        probs[i].score = 1e-8; // set to non-zero so that safety call will set to epsilon
    }
    CB_EXPLORE::safety(data.action_probs, min_prob * num_actions, /*zeros=*/false);
  }
  else
    CB_EXPLORE::safety(data.action_probs, min_prob * num_actions, !data.nounif);

  qsort((void*) probs.begin(), probs.size(), sizeof(action_score), reverse_order);
  for (size_t i = 0; i < num_actions; i++)
    preds[i] = probs[i];

  ++data.counter;
}

template <bool is_learn>
void predict_or_learn_softmax(cb_explore_adf& data, base_learner& base, v_array<example*>& examples)
{
  if (is_learn && test_adf_sequence(data.ec_seq) != nullptr)
    multiline_learn_or_predict<true>(base, examples, data.offset);
  else
    multiline_learn_or_predict<false>(base, examples, data.offset);

  v_array<action_score>& preds = examples[0]->pred.a_s;
  uint32_t num_actions = (uint32_t)preds.size();
  float norm = 0.;
  float max_score = preds[0].score;
  for (size_t i = 1; i < num_actions; i++)
    if (max_score < preds[i].score)
      max_score = preds[i].score;

  for (size_t i = 0; i < num_actions; i++)
  {
    float prob = exp(data.lambda*(preds[i].score - max_score));
    preds[i].score = prob;
    norm += prob;
  }
  for (size_t i = 0; i < num_actions; i++)
    preds[i].score /= norm;
  CB_EXPLORE::safety(preds, data.epsilon, true);
}

template <bool is_learn>
void predict_or_learn_agree(cb_explore_adf& data, base_learner& base, v_array<example*>& examples)
{
  std::vector<bool>& explored_actions = data.explored_actions;
  if (is_learn && test_adf_sequence(data.ec_seq) != nullptr)
  {
    // for active variant, impute loss = 1 for unexplored actions (not for MTR)
    if (data.nounifagree && data.gen_cs.cb_type != CB_TYPE_MTR
        && explored_actions.size() == examples[0]->pred.a_s.size())
    {
      uint32_t shared = static_cast<uint32_t>(CB::ec_is_example_header(*examples[0]));
      for (size_t i = shared; i < examples.size() - 1; ++i)
      {
        // cout << i - shared << ":" << examples[i]->l.cb.costs.size()
        //      << ":" << !explored_actions[i - shared] << " ";
        if (!explored_actions[i - shared])
        {
          CB::label& ld = examples[i]->l.cb;
          // cout << i - shared << " ";
          if (ld.costs.size() > 0)
          {
            cout << "cost: " << ld.costs[0].cost << endl;
            THROW("unexplored action has cb label!");
          }
          CB::cb_class cl;
          cl.action = 1 + i - shared;
          cl.cost = 1.f;
          cl.probability = 0;
          ld.costs.push_back(cl);
        }
      }
      // cout << "(imputed losses)" << &explored_actions << endl;
    }
    if (data.nounifagree)
      data.all->nounifagree = true;
    multiline_learn_or_predict<true>(base, examples, data.offset);
    if (data.nounifagree)
      data.all->nounifagree = false;
  }
  else
    multiline_learn_or_predict<false>(base, examples, data.offset);

  static std::vector<float> weights;
  if (!is_learn)
    get_disagree_weights(weights, data, base, examples);
  ++data.counter;
  ACTION_SCORE::action_scores& preds = examples[0]->pred.a_s;
  if (!is_learn)
  {
    uint32_t num_actions = (uint32_t)preds.size();
    const float et = data.agree_c0 * num_actions * log(data.counter) / (data.counter);
    const float threshold = sqrt(et) + et;
    for (size_t i = 0; i < preds.size(); ++i)
    { // cout << weights[preds[i].action] << " ";
      if (weights[preds[i].action] > threshold)
        preds[i].score = 0; // do not explore if disagreeing on this
      else
        preds[i].score = 1;
    }
    preds[0].score = 1; // always explore erm action
    // cout << "(weights)\n";

    CB_EXPLORE::safety(preds, 1.0, /*zeros=*/false);
  }
}

void end_examples(cb_explore_adf& data)
{
  if (data.need_to_clear)
    data.ec_seq.erase();
}

void finish(cb_explore_adf& data)
{
  data.ec_seq.delete_v();
  data.action_probs.delete_v();
  data.cs_labels.costs.delete_v();
  data.cs_labels_2.costs.delete_v();
  data.cb_labels.delete_v();
  for(size_t i = 0; i < data.prepped_cs_labels.size(); i++)
    data.prepped_cs_labels[i].costs.delete_v();
  data.prepped_cs_labels.delete_v();
  data.gen_cs.pred_scores.costs.delete_v();
}


//Semantics: Currently we compute the IPS loss no matter what flags
//are specified. We print the first action and probability, based on
//ordering by scores in the final output.

void output_example(vw& all, cb_explore_adf& c, example& ec, v_array<example*>* ec_seq)
{
  if (CB_ALGS::example_is_newline_not_header(ec)) return;

  size_t num_features = 0;

  float loss = 0.;
  ACTION_SCORE::action_scores preds = (*ec_seq)[0]->pred.a_s;

  for (size_t i = 0; i < (*ec_seq).size(); i++)
    if (!CB::ec_is_example_header(*(*ec_seq)[i]))
      num_features += (*ec_seq)[i]->num_features;

  bool is_test = false;
  if (c.gen_cs.known_cost.probability > 0)
  {
    for (uint32_t i = 0; i < preds.size(); i++)
    {
      float l = get_unbiased_cost(&c.gen_cs.known_cost, preds[i].action);
      loss += l*preds[i].score;
    }
  }
  else
    is_test = true;
  all.sd->update(ec.test_only, c.gen_cs.known_cost.probability > 0, loss, ec.weight, num_features);

  for (int sink : all.final_prediction_sink)
    print_action_score(sink, ec.pred.a_s, ec.tag);

  if (all.raw_prediction > 0)
  {
    string outputString;
    stringstream outputStringStream(outputString);
    v_array<CB::cb_class> costs = ec.l.cb.costs;

    for (size_t i = 0; i < costs.size(); i++)
    {
      if (i > 0) outputStringStream << ' ';
      outputStringStream << costs[i].action << ':' << costs[i].partial_prediction;
    }
    all.print_text(all.raw_prediction, outputStringStream.str(), ec.tag);
  }

  CB::print_update(all, is_test, ec, ec_seq, true);
}

void output_example_seq(vw& all, cb_explore_adf& data)
{
  if (data.ec_seq.size() > 0)
  {
    output_example(all, data, **(data.ec_seq.begin()), &(data.ec_seq));
    if (all.raw_prediction > 0)
      all.print_text(all.raw_prediction, "", data.ec_seq[0]->tag);
  }
}


void clear_seq_and_finish_examples(vw& all, cb_explore_adf& data)
{
  if (data.ec_seq.size() > 0)
    for (example* ecc : data.ec_seq)
      if (ecc->in_use)
        VW::finish_example(all, ecc);
  data.ec_seq.erase();
}

void finish_multiline_example(vw& all, cb_explore_adf& data, example& ec)
{
  if (data.need_to_clear)
  {
    if (data.ec_seq.size() > 0)
    {
      output_example_seq(all, data);
      CB_ADF::global_print_newline(all);
    }
    clear_seq_and_finish_examples(all, data);
    data.need_to_clear = false;
  }
}

template <bool is_learn>
void do_actual_learning(cb_explore_adf& data, base_learner& base)
{
  example* label_example=test_adf_sequence(data.ec_seq);
  data.gen_cs.known_cost = CB_ADF::get_observed_cost(data.ec_seq);

  if (label_example == nullptr || !is_learn)
  {
    if (label_example != nullptr)//extract label
    {
      data.action_label = label_example->l.cb;
      label_example->l.cb = data.empty_label;
    }
    switch (data.explore_type)
    {
    case EXPLORE_FIRST:
      predict_or_learn_first<false>(data, base, data.ec_seq);
      break;
    case EPS_GREEDY:
      predict_or_learn_greedy<false>(data, base, data.ec_seq);
      break;
    case SOFTMAX:
      predict_or_learn_softmax<false>(data, base, data.ec_seq);
      break;
    case BAG_EXPLORE:
      predict_or_learn_bag<false>(data, base, data.ec_seq);
      break;
    case COVER:
      predict_or_learn_cover<false>(data, base, data.ec_seq);
      break;
    case AGREE:
      predict_or_learn_agree<false>(data, base, data.ec_seq);
      break;
    case REGCB:
      predict_or_learn_regcb<false>(data, base, data.ec_seq);
      break;
    default:
      THROW("Unknown explorer type specified for contextual bandit learning: " << data.explore_type);
    }
    if (label_example != nullptr)	//restore label
      label_example->l.cb = data.action_label;
  }
  else
  {
    /*	v_array<float> temp_probs;
    temp_probs = v_init<float>();
    do_actual_learning<false>(data,base);
    for (size_t i = 0; i < data.ec_seq[0]->pred.a_s.size(); i++)
    temp_probs.push_back(data.ec_seq[0]->pred.a_s[i].score);*/

    switch (data.explore_type)
    {
    case EXPLORE_FIRST:
      predict_or_learn_first<is_learn>(data, base, data.ec_seq);
      break;
    case EPS_GREEDY:
      predict_or_learn_greedy<is_learn>(data, base, data.ec_seq);
      break;
    case SOFTMAX:
      predict_or_learn_softmax<is_learn>(data, base, data.ec_seq);
      break;
    case BAG_EXPLORE:
      predict_or_learn_bag<is_learn>(data, base, data.ec_seq);
      break;
    case COVER:
      predict_or_learn_cover<is_learn>(data, base, data.ec_seq);
      break;
    case AGREE:
      predict_or_learn_agree<is_learn>(data, base, data.ec_seq);
      break;
    case REGCB:
      predict_or_learn_regcb<is_learn>(data, base, data.ec_seq);
      break;
    default:
      THROW("Unknown explorer type specified for contextual bandit learning: " << data.explore_type);
    }

    /*	for (size_t i = 0; i < temp_probs.size(); i++)
      if (temp_probs[i] != data.ec_seq[0]->pred.a_s[i].score)
        cout << "problem! " << temp_probs[i] << " != " << data.ec_seq[0]->pred.a_s[i].score << " for " << data.ec_seq[0]->pred.a_s[i].action << endl;
        temp_probs.delete_v();*/
  }
}

template <bool is_learn>
void predict_or_learn(cb_explore_adf& data, base_learner& base, example &ec)
{
  vw* all = data.all;
  //data.base = &base;
  data.offset = ec.ft_offset;
  bool is_test_ec = CB::example_is_test(ec);
  bool need_to_break = VW::is_ring_example(*all, &ec) && (data.ec_seq.size() >= all->p->ring_size - 2);

  if ((CB_ALGS::example_is_newline_not_header(ec) && is_test_ec) || need_to_break)
  {
    data.ec_seq.push_back(&ec);
    if (data.ec_seq.size() == 1)
      cout << "Something is wrong---an example with no choice.  Do you have all 0 features? Or multiple empty lines?" << endl;
    else
      do_actual_learning<is_learn>(data, base);
    // using flag to clear, because ec_seq is used in finish_example
    data.need_to_clear = true;
  }
  else
  {
    if (data.need_to_clear)    // should only happen if we're NOT driving
    {
      data.ec_seq.erase();
      data.need_to_clear = false;
    }
    data.ec_seq.push_back(&ec);
  }
}

}


using namespace CB_EXPLORE_ADF;


base_learner* cb_explore_adf_setup(vw& all)
{
  //parse and set arguments
  if (missing_option(all, true, "cb_explore_adf", "Online explore-exploit for a contextual bandit problem with multiline action dependent features"))
    return nullptr;
  new_options(all, "CB_EXPLORE_ADF options")
  ("first", po::value<size_t>(), "tau-first exploration")
  ("epsilon", po::value<float>(), "epsilon-greedy exploration")
  ("bag", po::value<size_t>(), "bagging-based exploration")
  ("cover",po::value<size_t>() ,"Online cover based exploration")
  ("psi", po::value<float>(), "disagreement parameter for cover")
  ("agree", "Agreement exploration")
  ("agree_mellowness", po::value<float>(), "Agreement exploration threshold parameter c_0")
  ("nounif", "do not explore uniformly on zero-probability actions in cover")
  ("nounifagree", "do not explore uniformly on actions that d")
  ("regcb", "RegCB-elim exploration")
  ("regcbopt", "RegCB optimistic exploration")
  ("mellowness", po::value<float>(), "RegCB mellowness parameter c_0. Default 0.1")
  ("softmax", "softmax exploration")
  ("greedify", "always update first policy once in bagging")
  ("randomtie", "explore uniformly over random ties")
  ("lambda", po::value<float>(), "parameter for softmax");
  add_options(all);

  po::variables_map& vm = all.vm;
  cb_explore_adf& data = calloc_or_throw<cb_explore_adf>();

  data.all = &all;
  data.gen_cs.all = &all;
  if (count(all.args.begin(), all.args.end(), "--cb_adf") == 0)
    all.args.push_back("--cb_adf");

  all.delete_prediction = delete_action_scores;

  size_t problem_multiplier = 1;
  char type_string[10];

  data.randomtie = vm.count("randomtie") > 0;
  if (data.randomtie)
    *all.file_options << " --randomtie";
  if (vm.count("epsilon"))
  {
    data.epsilon = vm["epsilon"].as<float>();
    sprintf(type_string, "%f", data.epsilon);
    *all.file_options << " --epsilon "<<type_string;
  }
  if (vm.count("nounifagree"))
  { data.nounifagree = true;
    *all.file_options << " --nounifagree";
    // for notifying lower reductions
    // all.nounifagree = true;
  }
  data.agree_c0 = 0.1;
  if (vm.count("agree_mellowness"))
  { data.agree_c0 = vm["agree_mellowness"].as<float>();
    sprintf(type_string, "%f", data.agree_c0);
    *all.file_options << " --agree_mellowness "<<type_string;
  }
  if (vm.count("cover"))
  {
    data.cover_size = (uint32_t)vm["cover"].as<size_t>();
    data.explore_type = COVER;
    problem_multiplier = data.cover_size+1;
    *all.file_options << " --cover " << data.cover_size;

    data.psi = 1.0f;
    if (vm.count("psi"))
      data.psi = vm["psi"].as<float>();

    sprintf(type_string, "%f", data.psi);
    *all.file_options << " --psi " << type_string;
    if (vm.count("nounif"))
    {
      data.nounif = true;
      *all.file_options << " --nounif";
    }
    if (data.nounifagree && vm.count("agree_mellowness") == 0)
    { // c0 = psi by default unless explicitly provided
      data.agree_c0 = data.psi;
      sprintf(type_string, "%f", data.agree_c0);
      *all.file_options << " --agree_mellowness "<<type_string;
    }
  }
  else if (vm.count("bag"))
  {
    data.bag_size = (uint32_t)vm["bag"].as<size_t>();
    data.greedify = vm.count("greedify") > 0;
    data.explore_type = BAG_EXPLORE;
    problem_multiplier = data.bag_size + 1;
    *all.file_options << " --bag "<< data.bag_size;
    if (data.greedify)
      *all.file_options << " --greedify";
  }
  else if (vm.count("first"))
  {
    data.tau = (uint32_t)vm["first"].as<size_t>();
    data.explore_type = EXPLORE_FIRST;
    *all.file_options << " --first "<< data.tau;
  }
  else if (vm.count("softmax"))
  {
    data.lambda = 1.0;
    if (vm.count("lambda"))
      data.lambda = (float)vm["lambda"].as<float>();
    data.explore_type = SOFTMAX;
    sprintf(type_string, "%f", data.lambda);
    *all.file_options << " --softmax --lambda "<<type_string;
  }
  else if (vm.count("agree"))
  {
    data.explore_type = AGREE;
    *all.file_options << " --agree";
  }
  else if (vm.count("regcb") || vm.count("regcbopt"))
  {
    data.explore_type = REGCB;
    if (vm.count("regcbopt"))
      data.regcbopt = true;
    if (vm.count("mellowness"))
      data.c0 = (float)vm["mellowness"].as<float>();
    else
      data.c0 = 0.1;

    sprintf(type_string, "%f", data.c0);
    *all.file_options << (vm.count("regcbopt") ? " --regcbopt" : " --regcb")
                      << " --mellowness " << type_string;
  }
  else if (vm.count("epsilon"))
    data.explore_type = EPS_GREEDY;
  else //epsilon
  {
    data.epsilon = 0.05f;
    data.explore_type = EPS_GREEDY;
  }

  base_learner* base = setup_base(all);
  all.p->lp = CB::cb_label;
  all.label_type = label_type::cb;

  learner<cb_explore_adf>& l = init_learner(&data, base, CB_EXPLORE_ADF::predict_or_learn<true>, CB_EXPLORE_ADF::predict_or_learn<false>, problem_multiplier, prediction_type::action_probs);

  //Extract from lower level reductions.
  data.gen_cs.scorer = all.scorer;
  data.cs_ldf_learner = all.cost_sensitive;
  data.gen_cs.cb_type = CB_TYPE_IPS;
  if (all.vm.count("cb_type"))
  {
    std::string type_string;
    type_string = all.vm["cb_type"].as<std::string>();

    if (type_string.compare("dr") == 0)
      data.gen_cs.cb_type = CB_TYPE_DR;
    else if (type_string.compare("ips") == 0)
      data.gen_cs.cb_type = CB_TYPE_IPS;
    else if (type_string.compare("mtr") == 0)
    {
      if (vm.count("cover"))
        all.trace_message << "warning: currently, mtr is only used for the first policy in cover, other policies use dr" << endl;
      data.gen_cs.cb_type = CB_TYPE_MTR;
    }
    else
      all.trace_message << "warning: cb_type must be in {'ips','dr'}; resetting to ips." << std::endl;
  }

  if (data.explore_type == REGCB && data.gen_cs.cb_type != CB_TYPE_MTR)
  {
    all.trace_message << "warning: bad cb_type, RegCB only supports mtr!" << std::endl;
  }

  l.set_finish_example(CB_EXPLORE_ADF::finish_multiline_example);
  l.set_finish(CB_EXPLORE_ADF::finish);
  l.set_end_examples(CB_EXPLORE_ADF::end_examples);
  return make_base(l);
}

