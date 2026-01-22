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
@property
def num_steps(self) -> int:
    """Returns the number of steps to accumulate over"""
    return self.plugin_kwargs.get('num_steps', 1)