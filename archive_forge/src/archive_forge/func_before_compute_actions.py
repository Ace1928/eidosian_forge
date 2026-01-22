from gymnasium.spaces import Space
from typing import Dict, List, Optional, Union, TYPE_CHECKING
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.framework import try_import_torch, TensorType
from ray.rllib.utils.typing import LocalOptimizer, AlgorithmConfigDict
@DeveloperAPI
def before_compute_actions(self, *, timestep: Optional[Union[TensorType, int]]=None, explore: Optional[Union[TensorType, bool]]=None, tf_sess: Optional['tf.Session']=None, **kwargs):
    """Hook for preparations before policy.compute_actions() is called.

        Args:
            timestep: An optional timestep tensor.
            explore: An optional explore boolean flag.
            tf_sess: The tf-session object to use.
            **kwargs: Forward compatibility kwargs.
        """
    pass