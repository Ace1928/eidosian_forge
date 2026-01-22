import importlib.metadata
import operator
import re
import sys
from typing import Optional
from packaging import version
def require_version_core(requirement):
    """require_version wrapper which emits a core-specific hint on failure"""
    hint = "Try: `pip install transformers -U` or `pip install -e '.[dev]'` if you're working with git main"
    return require_version(requirement, hint)