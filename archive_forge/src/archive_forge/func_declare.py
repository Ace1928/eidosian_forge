import json
import pkgutil
import operator
from typing import List
from deprecated import deprecated
from deprecated.sphinx import versionadded
from lark import Lark, Transformer, v_args
import numpy as np
from pyquil.quilbase import (
from pyquil.quiltwaveforms import _wf_from_dict
from pyquil.quilatom import (
from pyquil.gates import (
@v_args(inline=True)
def declare(self, name, memory_type, memory_size, *sharing):
    shared, *offsets = sharing
    d = Declare(str(name), memory_type=str(memory_type), memory_size=int(memory_size) if memory_size else 1, shared_region=str(shared) if shared else None, offsets=offsets if shared else None)
    return d