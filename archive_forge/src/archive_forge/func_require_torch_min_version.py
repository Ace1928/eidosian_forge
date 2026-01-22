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
def require_torch_min_version(test_case=None, version=None):
    """
    Decorator marking that a test requires a particular torch version to be tested. These tests are skipped when an
    installed torch version is less than the required one.
    """
    if test_case is None:
        return partial(require_torch_min_version, version=version)
    return unittest.skipUnless(is_torch_version('>=', version), f'test requires torch version >= {version}')(test_case)