import functools
from collections import defaultdict
import numpy as np
import uuid
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from typing import Any, Dict, List, Optional, SupportsFloat, Union
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.utils import BufferWithInfiniteLookback
@property
def is_finalized(self) -> bool:
    """True, if the data in this episode is already stored as numpy arrays."""
    return self.rewards.finalized