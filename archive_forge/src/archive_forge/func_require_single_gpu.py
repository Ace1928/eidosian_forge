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
def require_single_gpu(test_case):
    """
    Decorator marking a test that requires CUDA on a single GPU. These tests are skipped when there are no GPU
    available or number of GPUs is more than one.
    """
    return unittest.skipUnless(torch.cuda.device_count() == 1, 'test requires a GPU')(test_case)