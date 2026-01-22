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
def require_trackers(test_case):
    """
    Decorator marking that a test requires at least one tracking library installed. These tests are skipped when none
    are installed
    """
    return unittest.skipUnless(_atleast_one_tracker_available, 'test requires at least one tracker to be available and for `comet_ml` to not be installed')(test_case)