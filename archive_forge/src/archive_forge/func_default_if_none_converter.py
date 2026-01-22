import typing
from ._compat import _AnnotationExtractor
from ._make import NOTHING, Factory, pipe
def default_if_none_converter(val):
    if val is not None:
        return val
    return default