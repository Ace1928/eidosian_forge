from __future__ import annotations
import argparse
import ast
import functools
import logging
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple
from flake8 import defaults
from flake8 import utils
from flake8._compat import FSTRING_END
from flake8._compat import FSTRING_MIDDLE
from flake8.plugins.finder import LoadedPlugin
def is_multiline_string(token: tokenize.TokenInfo) -> bool:
    """Check if this is a multiline string."""
    return token.type == FSTRING_END or (token.type == tokenize.STRING and '\n' in token.string)