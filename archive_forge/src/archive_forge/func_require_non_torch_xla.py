import asyncio
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import List, Union
from unittest import mock
import torch
import accelerate
from ..state import AcceleratorState, PartialState
from ..utils import (
def require_non_torch_xla(test_case):
    """
    Decorator marking a test as requiring an environment without TorchXLA. These tests are skipped when TorchXLA is
    available.
    """
    return unittest.skipUnless(not is_torch_xla_available(), 'test requires an env without TorchXLA')(test_case)