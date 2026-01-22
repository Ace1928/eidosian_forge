from __future__ import annotations
import ast
import builtins
import sys
import typing
from ast import (
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, overload
def get_memo_name(self) -> Name:
    if not self.memo_var_name:
        self.memo_var_name = Name(id='memo', ctx=Load())
    return self.memo_var_name