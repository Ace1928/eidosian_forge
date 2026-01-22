from typing import List, Optional, Union
import numpy as np
from cvxpy.atoms import (
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.constraints.exponential import OpRelEntrConeQuad
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
def norm_nuc_canon(expr: normNuc, real_args: List[Union[Expression, None]], imag_args: List[Union[Expression, None]], real2imag):
    """Canonicalize nuclear norm with Hermitian matrix input.
    """
    real, imag = hermitian_canon(expr, real_args, imag_args, real2imag)
    if imag_args[0] is not None:
        real /= 2
    return (real, imag)