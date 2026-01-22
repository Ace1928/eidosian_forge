import json
from collections import deque
from numbers import Number
from typing import Tuple, Optional
from ray.train._internal.checkpoint_manager import _CheckpointManager
from ray.tune.utils.serialization import TuneFunctionEncoder, TuneFunctionDecoder
def invalidate_cache(self):
    self._cached_json = None