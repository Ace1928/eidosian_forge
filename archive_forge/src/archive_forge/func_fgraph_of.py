import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def fgraph_of(*exprs):
    """ Transform SymPy expressions into Theano Computation.

    Parameters
    ==========
    exprs
        SymPy expressions

    Returns
    =======
    theano.gof.FunctionGraph
    """
    outs = list(map(theano_code_, exprs))
    ins = theano.gof.graph.inputs(outs)
    ins, outs = theano.gof.graph.clone(ins, outs)
    return theano.gof.FunctionGraph(ins, outs)