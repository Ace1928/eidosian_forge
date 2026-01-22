from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def find_indices_left_equal(self, threshold: int, shift: bool=0):
    indices = []
    for num in self:
        if num > threshold or self.index(num) + shift < 0:
            break
        else:
            indices.append(self.index(num))
    return indices