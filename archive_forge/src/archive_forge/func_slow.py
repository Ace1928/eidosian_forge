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
def slow(test_case):
    """
    Decorator marking a test as slow. Slow tests are skipped by default. Set the RUN_SLOW environment variable to a
    truthy value to run them.
    """
    return unittest.skipUnless(_run_slow_tests, 'test is slow')(test_case)