import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.utils.annotations import Deprecated, DeveloperAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.utils.typing import (
from ray.util import log_once
@DeveloperAPI
def last_terminated_for(self, agent_id: AgentID=_DUMMY_AGENT_ID) -> bool:
    """Returns the last `terminated` flag for the specified AgentID.

        Args:
            agent_id: The agent's ID to get the last `terminated` flag for.

        Returns:
            Last terminated flag for the specified AgentID.
        """
    if agent_id not in self._agent_to_last_terminated:
        self._agent_to_last_terminated[agent_id] = False
    return self._agent_to_last_terminated[agent_id]