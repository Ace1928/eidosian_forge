import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
def get_parametric_distribution_for_action_space(action_space, continuous_config=None):
    """Returns an action distribution parametrization based on the action space.

  Args:
    action_space: action space of the environment
    continuous_config: Configuration for the continuous action distribution
      (used when needed by the action space)..
  """
    if isinstance(action_space, gym.spaces.Discrete):
        return categorical_distribution(action_space.n, dtype=action_space.dtype)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        check_multi_discrete_space(action_space)
        return multi_categorical_distribution(n_dimensions=len(action_space.nvec), n_actions_per_dim=action_space.nvec[0], dtype=action_space.dtype)
    elif isinstance(action_space, gym.spaces.Box):
        check_box_space(action_space)
        if continuous_config is None:
            continuous_config = ContinuousDistributionConfig()
        if continuous_config.postprocessor == 'Tanh':
            return normal_tanh_distribution(num_actions=action_space.shape[0], gaussian_std_fn=continuous_config.gaussian_std_fn)
        elif continuous_config.postprocessor == 'ClippedIdentity':
            return normal_clipped_distribution(num_actions=action_space.shape[0], gaussian_std_fn=continuous_config.gaussian_std_fn)
        else:
            raise ValueError(f'Postprocessor {continuous_config.postprocessor} not supported.')
    elif isinstance(action_space, gym.spaces.Tuple):
        return joint_distribution([get_parametric_distribution_for_action_space(subspace, continuous_config) for subspace in action_space])
    else:
        raise ValueError(f'Unsupported action space {action_space}')