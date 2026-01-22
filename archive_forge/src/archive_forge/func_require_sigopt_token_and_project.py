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
def require_sigopt_token_and_project(test_case):
    """
    Decorator marking a test that requires sigopt API token.
    """
    use_auth_token = os.environ.get('SIGOPT_API_TOKEN', None)
    has_sigopt_project = os.environ.get('SIGOPT_PROJECT', None)
    if use_auth_token is None or has_sigopt_project is None:
        return unittest.skip('test requires an environment variable `SIGOPT_API_TOKEN` and `SIGOPT_PROJECT`')(test_case)
    else:
        return test_case