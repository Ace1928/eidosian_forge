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
def names_from_assignments(assign_target):
    if isinstance(assign_target, ast.Name):
        yield assign_target.id
    elif isinstance(assign_target, ast.Starred):
        yield from names_from_assignments(assign_target.value)
    elif isinstance(assign_target, (ast.List, ast.Tuple)):
        for child in assign_target.elts:
            yield from names_from_assignments(child)