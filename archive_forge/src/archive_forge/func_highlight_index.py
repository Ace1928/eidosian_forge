import ast
import itertools
import types
from collections import OrderedDict, Counter, defaultdict
from types import FrameType, TracebackType
from typing import (
from asttokens import ASTText
def highlight_index(f):
    try:
        i = f()
    except ValueError:
        return None
    highlighted[i] = True
    return i