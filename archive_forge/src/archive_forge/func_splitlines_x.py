import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def splitlines_x(s):
    if not s:
        return []
    else:
        return s.splitlines()