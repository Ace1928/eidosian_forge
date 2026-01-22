from recsim import choice_model
from recsim.environments import (
from ray.rllib.env.wrappers.recsim import make_recsim_env
from ray.tune import register_env
Examples for RecSim envs ready to be used by RLlib Algorithms.

RecSim is a configurable recommender systems simulation platform.
Source: https://github.com/google-research/recsim
