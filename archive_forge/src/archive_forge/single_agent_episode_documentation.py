import functools
from collections import defaultdict
import numpy as np
import uuid
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from typing import Any, Dict, List, Optional, SupportsFloat, Union
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.utils import BufferWithInfiniteLookback
Enable squared bracket indexing- and slicing syntax, e.g. episode[-4:].