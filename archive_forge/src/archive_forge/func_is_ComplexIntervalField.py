from ..sage_helper import sage_method, _within_sage
from ..number import Number
from . import verifyHyperbolicity
def is_ComplexIntervalField(z):
    return isinstance(z, sage.rings.abc.ComplexIntervalField)