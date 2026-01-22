from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@noKwargs
@noPosargs
def to_upper_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
    return self.held_object.upper()