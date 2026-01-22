from __future__ import annotations
import base64
import dataclasses
import hashlib
import inspect
import operator
import platform
import sys
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
def formatargandannotation(arg):
    result = formatarg(arg)
    if arg in annotations:
        result += ': ' + formatannotation(annotations[arg])
    return result