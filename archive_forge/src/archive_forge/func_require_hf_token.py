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
def require_hf_token(test_case):
    """
    Decorator marking a test that requires huggingface hub token.
    """
    use_auth_token = os.environ.get('HF_AUTH_TOKEN', None)
    if use_auth_token is None:
        return unittest.skip('test requires hf token as `HF_AUTH_TOKEN` environment variable')(test_case)
    else:
        return test_case