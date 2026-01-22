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
def check_for_b028(self, node):
    if isinstance(node.func, ast.Attribute) and node.func.attr == 'warn' and isinstance(node.func.value, ast.Name) and (node.func.value.id == 'warnings') and (not any((kw.arg == 'stacklevel' for kw in node.keywords))) and (len(node.args) < 3):
        self.errors.append(B028(node.lineno, node.col_offset))