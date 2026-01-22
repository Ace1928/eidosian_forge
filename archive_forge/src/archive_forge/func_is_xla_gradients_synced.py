from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
@is_xla_gradients_synced.setter
def is_xla_gradients_synced(self, is_synced):
    """Set the _is_xla_gradients_synced attribute."""
    self._is_xla_gradients_synced = is_synced