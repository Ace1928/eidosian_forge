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
def require_multi_device(test_case):
    """
    Decorator marking a test that requires a multi-device setup. These tests are skipped on a machine without multiple
    devices.
    """
    return unittest.skipUnless(device_count > 1, 'test requires multiple hardware accelerators')(test_case)