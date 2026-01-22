import os
import logging
from typing import Optional, List, Tuple
from functools import lru_cache
from importlib.util import find_spec
from ray._private.accelerators.accelerator import AcceleratorManager
Attempt to detect the HPU family type.
        Returns:
            The device name (GAUDI, GAUDI2) if detected else None.
        