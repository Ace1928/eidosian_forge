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
def def_calibration(self, name, params, qubits, instructions):
    for p in params:
        mrefs = _contained_mrefs(p)
        if mrefs:
            raise ValueError(f"Unexpected memory references {mrefs} in DEFCAL {name}. Did you forget a '%'?")
    dc = DefCalibration(name, params, qubits, instructions)
    return dc