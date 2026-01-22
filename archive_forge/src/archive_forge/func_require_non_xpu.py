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
def require_non_xpu(test_case):
    """
    Decorator marking a test that should be skipped for XPU.
    """
    return unittest.skipUnless(torch_device != 'xpu', 'test requires a non-XPU')(test_case)