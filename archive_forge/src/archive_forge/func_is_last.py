from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from xformers.components import NormalizationType, ResidualNormStyle
from xformers.components.feedforward import FEEDFORWARD_REGISTRY, FeedforwardConfig
from xformers.components.positional_embedding import (
from xformers.utils import generate_matching_config
def is_last(self):
    return bool(self.bitmask & LayerPositionBitmask.Last)