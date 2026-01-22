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
def require_clearml(test_case):
    """
    Decorator marking a test that requires clearml installed. These tests are skipped when clearml isn't installed
    """
    return unittest.skipUnless(is_clearml_available(), 'test requires clearml')(test_case)