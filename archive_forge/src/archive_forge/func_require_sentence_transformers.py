import importlib.util
import itertools
import os
import subprocess
import sys
import unittest
from collections.abc import MutableMapping
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
import torch
from . import (
def require_sentence_transformers(test_case):
    return unittest.skipUnless(is_sentence_transformers_available(), 'test requires sentence-transformers')(test_case)