from typing import List, Optional, Union
import numpy as np
from cvxpy.atoms import (
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.constraints.exponential import OpRelEntrConeQuad
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
def op_rel_entr_cone_canon(expr: OpRelEntrConeQuad, real_args: List[Union[Expression, None]], imag_args: List[Union[Expression, None]], real2imag):
    """Transform Hermitian input for OpRelEntrConeQuad into equivalent
    symmetric input for OpRelEntrConeQuad
    """
    must_expand = any((a is not None for a in imag_args))
    if must_expand:
        X_dilation = expand_complex(real_args[0], imag_args[0])
        Y_dilation = expand_complex(real_args[1], imag_args[1])
        Z_dilation = expand_complex(real_args[2], imag_args[2])
        canon_expr = expr.copy([X_dilation, Y_dilation, Z_dilation])
    else:
        canon_expr = expr.copy(real_args)
    return ([canon_expr], None)