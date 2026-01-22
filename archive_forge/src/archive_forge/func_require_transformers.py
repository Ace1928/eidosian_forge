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
def require_transformers(test_case):
    """
    Decorator marking a test that requires transformers. These tests are skipped when they are not.
    """
    return unittest.skipUnless(is_transformers_available(), 'test requires the transformers library')(test_case)