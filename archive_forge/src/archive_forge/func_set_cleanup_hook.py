import dataclasses
import inspect
from typing import Callable, Dict, List, Optional
import torch._C
from torch._guards import Guard
from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..device_interface import get_interface_for_device
from ..exc import unimplemented, Unsupported
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalStateSource
from .base import VariableTracker
from .functions import (
def set_cleanup_hook(self, tx, fn=None):
    if fn is None:

        def fn():
            self._call_func(tx, self.initial_values)
    self.state.cleanup_fn = fn
    tx.output.add_cleanup_hook(self.state.cleanup)