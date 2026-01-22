import re
from itertools import product
import numpy as np
import copy
from typing import (
from pyquil.quilatom import (
from .quil import Program
from .gates import H, RZ, RX, CNOT, X, PHASE, QUANTUM_GATES
from numbers import Number, Complex
from collections import OrderedDict
import warnings
def exp_wrap(param: float) -> Program:
    prog = Program()
    if is_identity(term):
        prog.inst(X(0))
        prog.inst(PHASE(-param * coeff, 0))
        prog.inst(X(0))
        prog.inst(PHASE(-param * coeff, 0))
    elif is_zero(term):
        pass
    else:
        prog += _exponentiate_general_case(term, param)
    return prog