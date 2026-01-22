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
def check_for_b021(self, node):
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.JoinedStr):
        self.errors.append(B021(node.body[0].value.lineno, node.body[0].value.col_offset))