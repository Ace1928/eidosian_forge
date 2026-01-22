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
def require_mlu(test_case):
    """
    Decorator marking a test that requires MLU. These tests are skipped when there are no MLU available.
    """
    return unittest.skipUnless(is_mlu_available(), 'test require a MLU')(test_case)