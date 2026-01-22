import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def squareroot(attrs, inputs, proto_obj):
    """Returns element-wise square-root value of the input."""
    return ('sqrt', attrs, inputs)