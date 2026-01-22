from __future__ import annotations
import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union
import torch
from .common import amp_definitely_not_available
def set_growth_factor(self, new_factor: float) -> None:
    """Set a new scale growth factor.

        Args:
            new_scale (float):  Value to use as the new scale growth factor.
        """
    self._growth_factor = new_factor