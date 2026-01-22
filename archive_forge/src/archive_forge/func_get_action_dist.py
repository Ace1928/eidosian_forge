from functools import partial
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
import logging
import numpy as np
import tree  # pip install dm_tree
from typing import List, Optional, Type, Union
from ray.tune.registry import (
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.typing import ModelConfigDict, TensorType
@staticmethod
@DeveloperAPI
def get_action_dist(action_space: gym.Space, config: ModelConfigDict, dist_type: Optional[Union[str, Type[ActionDistribution]]]=None, framework: str='tf', **kwargs) -> (type, int):
    """Returns a distribution class and size for the given action space.

        Args:
            action_space: Action space of the target gym env.
            config (Optional[dict]): Optional model config.
            dist_type (Optional[Union[str, Type[ActionDistribution]]]):
                Identifier of the action distribution (str) interpreted as a
                hint or the actual ActionDistribution class to use.
            framework: One of "tf2", "tf", "torch", or "jax".
            kwargs: Optional kwargs to pass on to the Distribution's
                constructor.

        Returns:
            Tuple:
                - dist_class (ActionDistribution): Python class of the
                    distribution.
                - dist_dim (int): The size of the input vector to the
                    distribution.
        """
    dist_cls = None
    config = config or MODEL_DEFAULTS
    if config.get('custom_action_dist'):
        custom_action_config = config.copy()
        action_dist_name = custom_action_config.pop('custom_action_dist')
        logger.debug('Using custom action distribution {}'.format(action_dist_name))
        dist_cls = _global_registry.get(RLLIB_ACTION_DIST, action_dist_name)
        return ModelCatalog._get_multi_action_distribution(dist_cls, action_space, custom_action_config, framework)
    elif type(dist_type) is type and issubclass(dist_type, ActionDistribution) and (dist_type not in (MultiActionDistribution, TorchMultiActionDistribution)):
        dist_cls = dist_type
    elif isinstance(action_space, Box):
        if action_space.dtype.name.startswith('int'):
            low_ = np.min(action_space.low)
            high_ = np.max(action_space.high)
            dist_cls = TorchMultiCategorical if framework == 'torch' else MultiCategorical
            num_cats = int(np.product(action_space.shape))
            return (partial(dist_cls, input_lens=[high_ - low_ + 1 for _ in range(num_cats)], action_space=action_space), num_cats * (high_ - low_ + 1))
        else:
            if len(action_space.shape) > 1:
                raise UnsupportedSpaceException('Action space has multiple dimensions {}. '.format(action_space.shape) + 'Consider reshaping this into a single dimension, using a custom action distribution, using a Tuple action space, or the multi-agent API.')
            if dist_type is None:
                return (partial(TorchDiagGaussian if framework == 'torch' else DiagGaussian, action_space=action_space), DiagGaussian.required_model_output_shape(action_space, config))
            elif dist_type == 'deterministic':
                dist_cls = TorchDeterministic if framework == 'torch' else Deterministic
    elif isinstance(action_space, Discrete):
        if framework == 'torch':
            dist_cls = TorchCategorical
        elif framework == 'jax':
            from ray.rllib.models.jax.jax_action_dist import JAXCategorical
            dist_cls = JAXCategorical
        else:
            dist_cls = Categorical
    elif dist_type in (MultiActionDistribution, TorchMultiActionDistribution) or isinstance(action_space, (Tuple, Dict)):
        return ModelCatalog._get_multi_action_distribution(MultiActionDistribution if framework == 'tf' else TorchMultiActionDistribution, action_space, config, framework)
    elif isinstance(action_space, Simplex):
        dist_cls = TorchDirichlet if framework == 'torch' else Dirichlet
    elif isinstance(action_space, MultiDiscrete):
        dist_cls = TorchMultiCategorical if framework == 'torch' else MultiCategorical
        return (partial(dist_cls, input_lens=action_space.nvec), int(sum(action_space.nvec)))
    else:
        raise NotImplementedError('Unsupported args: {} {}'.format(action_space, dist_type))
    return (dist_cls, int(dist_cls.required_model_output_shape(action_space, config)))