import argparse
import os
from pettingzoo.classic import rps_v2
import random
import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import (
from ray.rllib.env import PettingZooEnv
from ray.rllib.examples.policy.rock_paper_scissors_dummies import (
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env
def run_heuristic_vs_learned(args, use_lstm=False, algorithm_config=None):
    """Run heuristic policies vs a learned agent.

    The learned agent should eventually reach a reward of ~5 with
    use_lstm=False, and ~7 with use_lstm=True. The reason the LSTM policy
    can perform better is since it can distinguish between the always_same vs
    beat_last heuristics.
    """

    def select_policy(agent_id, episode, **kwargs):
        if agent_id == 'player_0':
            return 'learned'
        else:
            return random.choice(['always_same', 'beat_last'])
    config = (algorithm_config or PPOConfig()).environment('RockPaperScissors').framework(args.framework).rollouts(num_rollout_workers=0, num_envs_per_worker=4).multi_agent(policies={'always_same': PolicySpec(policy_class=AlwaysSameHeuristic), 'beat_last': PolicySpec(policy_class=BeatLastHeuristic), 'learned': PolicySpec(config=AlgorithmConfig.overrides(model={'use_lstm': use_lstm}, framework_str=args.framework))}, policy_mapping_fn=select_policy, policies_to_train=['learned']).reporting(metrics_num_episodes_for_smoothing=200).resources(num_gpus=int(os.environ.get('RLLIB_NUM_GPUS', '0')))
    algo = config.build()
    reward_diff = 0
    for _ in range(args.stop_iters):
        results = algo.train()
        if 'policy_always_same_reward' not in results['hist_stats']:
            reward_diff = 0
            continue
        reward_diff = sum(results['hist_stats']['policy_learned_reward'])
        print(f'delta_r={reward_diff}')
        if results['timesteps_total'] > args.stop_timesteps:
            break
        elif reward_diff > args.stop_reward:
            return
    if args.as_test:
        raise ValueError('Desired reward difference ({}) not reached! Only got to {}.'.format(args.stop_reward, reward_diff))