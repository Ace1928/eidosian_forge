import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
def count_field_parse_action(s, l, t):
    nonlocal array_expr
    n = t[0]
    array_expr <<= expr * n if n else Empty()
    del t[:]