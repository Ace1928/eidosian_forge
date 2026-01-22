from collections import OrderedDict
import gymnasium as gym
from typing import Dict, List, Optional
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID
Wrap an existing MultiAgentEnv to group agent ID together.

        See `MultiAgentEnv.with_agent_groups()` for more detailed usage info.

        Args:
            env: The env to wrap and whose agent IDs to group into new agents.
            groups: Mapping from group id to a list of the agent ids
                of group members. If an agent id is not present in any group
                value, it will be left ungrouped. The group id becomes a new agent ID
                in the final environment.
            obs_space: Optional observation space for the grouped
                env. Must be a tuple space. If not provided, will infer this to be a
                Tuple of n individual agents spaces (n=num agents in a group).
            act_space: Optional action space for the grouped env.
                Must be a tuple space. If not provided, will infer this to be a Tuple
                of n individual agents spaces (n=num agents in a group).
        