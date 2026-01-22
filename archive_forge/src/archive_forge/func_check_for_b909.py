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
def check_for_b909(self, node: ast.For):
    if isinstance(node.iter, ast.Name):
        name = _to_name_str(node.iter)
    elif isinstance(node.iter, ast.Attribute):
        name = _to_name_str(node.iter)
    else:
        return
    checker = B909Checker(name)
    checker.visit(node.body)
    for mutation in checker.mutations:
        self.errors.append(B909(mutation.lineno, mutation.col_offset))