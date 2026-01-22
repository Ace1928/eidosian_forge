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
def check_for_b015(self, node):
    if isinstance(self.node_stack[-2], ast.Expr):
        self.errors.append(B015(node.lineno, node.col_offset))