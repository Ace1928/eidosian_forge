import random
from collections import defaultdict
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.collectors.simple_list_collector import (
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvInfoDict, PolicyID, TensorType
@DeveloperAPI
def policy_for(self, agent_id: AgentID=_DUMMY_AGENT_ID, refresh: bool=False) -> PolicyID:
    """Returns and stores the policy ID for the specified agent.

        If the agent is new, the policy mapping fn will be called to bind the
        agent to a policy for the duration of the entire episode (even if the
        policy_mapping_fn is changed in the meantime!).

        Args:
            agent_id: The agent ID to lookup the policy ID for.

        Returns:
            The policy ID for the specified agent.
        """
    if agent_id not in self._agent_to_policy or refresh:
        policy_id = self._agent_to_policy[agent_id] = self.policy_mapping_fn(agent_id, self, worker=self.worker)
    else:
        policy_id = self._agent_to_policy[agent_id]
    if policy_id not in self.policy_map:
        raise KeyError(f"policy_mapping_fn returned invalid policy id '{policy_id}'!")
    return policy_id