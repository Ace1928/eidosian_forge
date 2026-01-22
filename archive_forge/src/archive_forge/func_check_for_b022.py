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
def check_for_b022(self, node):
    item = node.items[0]
    item_context = item.context_expr
    if hasattr(item_context, 'func') and hasattr(item_context.func, 'value') and hasattr(item_context.func.value, 'id') and (item_context.func.value.id == 'contextlib') and hasattr(item_context.func, 'attr') and (item_context.func.attr == 'suppress') and (len(item_context.args) == 0):
        self.errors.append(B022(node.lineno, node.col_offset))