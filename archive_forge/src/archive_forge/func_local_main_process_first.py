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
@contextmanager
def local_main_process_first(self):
    """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.
        """
    with PartialState().local_main_process_first():
        yield