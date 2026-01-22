import collections
import logging
import numpy as np
from typing import List, Any, Dict, Optional, TYPE_CHECKING
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.typing import PolicyID, AgentID
from ray.util.debug import log_once
def check_missing_dones(self) -> None:
    for agent_id, builder in self.agent_builders.items():
        if not builder.buffers.is_terminated_or_truncated():
            raise ValueError("The environment terminated for all agents, but we still don't have a last observation for agent {} (policy {}). ".format(agent_id, self.agent_to_policy[agent_id]) + "Please ensure that you include the last observations of all live agents when setting '__all__' terminated|truncated to True. ")