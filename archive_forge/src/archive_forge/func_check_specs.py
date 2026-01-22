from pythran.tables import MODULES
from pythran.intrinsic import Class
from pythran.typing import Tuple, List, Set, Dict
from pythran.utils import isstr
from pythran import metadata
import beniget
import gast as ast
import logging
import numpy as np
def check_specs(specs, types):
    """
    Does nothing but raising PythranSyntaxError if specs
    are incompatible with the actual code
    """
    from pythran.types.tog import unify, clone, tr
    from pythran.types.tog import Function, TypeVariable, InferenceError
    for fname, signatures in specs.functions.items():
        ftype = types[fname]
        for signature in signatures:
            sig_type = Function([tr(p) for p in signature], TypeVariable())
            try:
                unify(clone(sig_type), clone(ftype))
            except InferenceError:
                raise PythranSyntaxError('Specification for `{}` does not match inferred type:\nexpected `{}`\ngot `Callable[[{}], ...]`'.format(fname, ftype, ', '.join(map(str, sig_type.types[:-1]))))