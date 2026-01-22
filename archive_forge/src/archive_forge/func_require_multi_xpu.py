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
def require_multi_xpu(test_case):
    """
    Decorator marking a test that requires a multi-XPU setup. These tests are skipped on a machine without multiple
    XPUs.
    """
    return unittest.skipUnless(torch.xpu.device_count() > 1, 'test requires multiple XPUs')(test_case)