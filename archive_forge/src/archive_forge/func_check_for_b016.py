from __future__ import annotations
import ast
import builtins
import itertools
import logging
import math
import re
import sys
import warnings
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from keyword import iskeyword
from typing import Dict, List, Set, Union
import attr
import pycodestyle
def check_for_b016(self, node):
    if isinstance(node.exc, ast.JoinedStr) or (isinstance(node.exc, ast.Constant) and (isinstance(node.exc.value, (int, float, complex, str, bool)) or node.exc.value is None)):
        self.errors.append(B016(node.lineno, node.col_offset))