from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@noKwargs
@noPosargs
@FeatureNew('str.splitlines', '1.2.0')
def splitlines_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> T.List[str]:
    return self.held_object.splitlines()