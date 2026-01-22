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
def check_for_b908(self, node: ast.With):
    if len(node.body) < 2:
        return
    for node_item in node.items:
        if self._is_assertRaises_like(node_item):
            self.errors.append(B908(node.lineno, node.col_offset))