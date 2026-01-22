import importlib.util
import itertools
import os
import subprocess
import sys
import unittest
from collections.abc import MutableMapping
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
import torch
from . import (
def require_ort_training(test_case):
    """
    Decorator marking a test that requires onnxruntime-training and torch_ort correctly installed and configured.
    These tests are skipped otherwise.
    """
    return unittest.skipUnless(is_ort_training_available(), 'test requires torch_ort correctly installed and configured')(test_case)