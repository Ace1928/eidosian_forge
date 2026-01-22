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
def children_in_scope(node):
    yield node
    if not isinstance(node, FUNCTION_NODES):
        for child in ast.iter_child_nodes(node):
            yield from children_in_scope(child)