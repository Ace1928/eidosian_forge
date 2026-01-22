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
def check_for_b005(self, node):
    if isinstance(node, ast.Import):
        for name in node.names:
            self._b005_imports.add(name.asname or name.name)
    elif isinstance(node, ast.ImportFrom):
        for name in node.names:
            self._b005_imports.add(f'{node.module}.{name.name or name.asname}')
    elif isinstance(node, ast.Call):
        if node.func.attr not in B005.methods:
            return
        if isinstance(node.func.value, ast.Name) and node.func.value.id in self._b005_imports:
            return
        if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or (not isinstance(node.args[0].value, str)):
            return
        call_path = '.'.join(compose_call_path(node.func.value))
        if call_path in B005.valid_paths:
            return
        value = node.args[0].value
        if len(value) == 1:
            return
        if len(value) == len(set(value)):
            return
        self.errors.append(B005(node.lineno, node.col_offset))