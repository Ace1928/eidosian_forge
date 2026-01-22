import argparse
import gymnasium as gym
import os
import ray
from ray.rllib.algorithms.dqn import DQNConfig, DQNTFPolicy, DQNTorchPolicy
from ray.rllib.algorithms.ppo import (
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
Example of using two different training methods at once in multi-agent.

Here we create a number of CartPole agents, some of which are trained with
DQN, and some of which are trained with PPO. We periodically sync weights
between the two algorithms (note that no such syncing is needed when using just
a single training method).

For a simpler example, see also: multiagent_cartpole.py
