import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def unify_int_or_none(t, name):
    try:
        unify(t, Integer())
    except InferenceError:
        try:
            unify(t, NoneType)
        except InferenceError:
            raise PythranTypeError('Invalid slice {} type `{}`, expecting int or None'.format(name, t))