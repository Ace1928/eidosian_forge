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
def is_abc_class(value, name='ABC'):
    if isinstance(value, ast.keyword):
        return value.arg == 'metaclass' and is_abc_class(value.value, 'ABCMeta')
    return isinstance(value, ast.Name) and value.id == name or (isinstance(value, ast.Attribute) and value.attr == name and isinstance(value.value, ast.Name) and (value.value.id == 'abc'))