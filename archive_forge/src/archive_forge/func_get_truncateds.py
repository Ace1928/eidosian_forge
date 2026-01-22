from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def get_truncateds(self) -> MultiAgentDict:
    truncateds = {agent_id: self.agent_episodes[agent_id].is_truncated for agent_id in self._agent_ids}
    truncateds.update({'__all__': self.is_terminated})
    return truncateds