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
def last_extra_action_outs_for(self, agent_id: AgentID=_DUMMY_AGENT_ID) -> dict:
    """Returns the last extra-action outputs for the specified agent.

        This data is returned by a call to
        `Policy.compute_actions_from_input_dict` as the 3rd return value
        (1st return value = action; 2nd return value = RNN state outs).

        Args:
            agent_id: The agent's ID to get the last extra-action outs for.

        Returns:
            The last extra-action outs for the specified AgentID.
        """
    return self._agent_to_last_extra_action_outs[agent_id]