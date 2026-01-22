import logging
from typing import Callable, Tuple, Optional, List, Dict, Any, TYPE_CHECKING, Union, Set
import gymnasium as gym
import ray
from ray.rllib.utils.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvType, MultiEnvDict
@PublicAPI
def last(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
    """Returns the last observations, rewards, done- truncated flags and infos ...

        that were returned by the environment.

        Returns:
            The last observations, rewards, done- and truncated flags, and infos
            for each sub-environment.
        """
    logger.warning('last has not been implemented for this environment.')
    return ({}, {}, {}, {}, {})