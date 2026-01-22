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
def prev_reward_for(self, agent_id: AgentID=_DUMMY_AGENT_ID) -> float:
    """Returns the previous reward for the specified agent, or zero.

        The "previous" reward is the one received one timestep before the
        most recently received reward of the agent.

        Args:
            agent_id: The agent's ID to get the previous reward for.

        Returns:
            Previous reward for the the specified AgentID.
            Zero in case the agent has never performed any actions (or only
            one) in the episode.
        """
    history = self._agent_reward_history[agent_id]
    if len(history) >= 2:
        return history[-2]
    else:
        return 0.0