import argparse
import os
import sys
import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.utils import try_import_pyspiel, try_import_open_spiel
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune import CLIReporter, register_env
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.examples.rl_module.random_rl_module import RandomRLModule
from open_spiel.python.rl_environment import Environment  # noqa: E402
def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['tf', 'tf2', 'torch'], default='torch', help='The DL framework specifier.')
    parser.add_argument('--num-cpus', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--from-checkpoint', type=str, default=None, help='Full path to a checkpoint file for restoring a previously saved Algorithm state.')
    parser.add_argument('--env', type=str, default='connect_four', choices=['markov_soccer', 'connect_four'])
    parser.add_argument('--stop-iters', type=int, default=200, help='Number of iterations to train.')
    parser.add_argument('--stop-timesteps', type=int, default=10000000, help='Number of timesteps to train.')
    parser.add_argument('--win-rate-threshold', type=float, default=0.95, help="Win-rate at which we setup another opponent by freezing the current main policy and playing against a uniform distribution of previously frozen 'main's from here on.")
    parser.add_argument('--num-episodes-human-play', type=int, default=10, help='How many episodes to play against the user on the command line after training has finished.')
    parser.add_argument('--as-test', action='store_true', help='Whether this script should be run as a test: --stop-reward must be achieved within --stop-timesteps AND --stop-iters.')
    parser.add_argument('--min-win-rate', type=float, default=0.5, help='Minimum win rate to consider the test passed.')
    args = parser.parse_args()
    print(f'Running with following CLI args: {args}')
    return args