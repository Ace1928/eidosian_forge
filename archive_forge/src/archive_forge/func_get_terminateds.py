from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def get_terminateds(self) -> MultiAgentDict:
    """Gets the terminateds at given indices."""
    terminateds = {agent_id: self.agent_episodes[agent_id].is_terminated for agent_id in self._agent_ids}
    terminateds.update({'__all__': self.is_terminated})
    return terminateds