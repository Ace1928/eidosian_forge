import os
import sys
import tempfile
import torch
from .state import AcceleratorState, PartialState
from .utils import (
def test_launch():
    """Verify a `PartialState` can be initialized."""
    _ = PartialState()