from __future__ import annotations
import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from enum import auto
from functools import reduce
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
from sqlglot.tokens import Token
def else_(self, condition: ExpOrStr, copy: bool=True, **opts) -> Case:
    instance = maybe_copy(self, copy)
    instance.set('default', maybe_parse(condition, copy=copy, **opts))
    return instance