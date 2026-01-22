import array
import contextlib
import os
import types
import typing
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.embedded
import rpy2.rinterface_lib.openrlib
import rpy2.rlike.container as rlc
from rpy2.robjects.robject import RObjectMixin, RObject
import rpy2.robjects.functions
from rpy2.robjects.environments import (Environment,
from rpy2.robjects.methods import methods_env
from rpy2.robjects.methods import RS4
from . import conversion
from . import vectors
from . import language
from rpy2.rinterface import (Sexp,
from rpy2.robjects.functions import Function
from rpy2.robjects.functions import SignatureTranslatedFunction
def sequence_to_vector(lst):
    curr_typeorder = -1
    i = None
    for i, elt in enumerate(lst):
        cls = type(elt)
        if cls in TYPEORDER:
            if TYPEORDER[cls][0] > curr_typeorder:
                curr_typeorder, curr_type = TYPEORDER[cls]
        else:
            raise ValueError('The element %i in the list has a type that cannot be handled.' % i)
    if i is None:
        raise ValueError('The parameter "lst" is an empty sequence. The type of the corresponding R vector cannot be determined.')
    res = curr_type(lst)
    return res