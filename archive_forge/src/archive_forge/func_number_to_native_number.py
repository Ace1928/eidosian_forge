from .sage_helper import _within_sage
from .pari import *
import re
def number_to_native_number(n):
    """
            Simply returns the given SnapPy number.

            In general snappy.number.number_to_native_number converts a SnapPy number to
            the corresponding SageMath type (when in SageMath) or just returns
            the SnapPy number itself (when SageMath is not available).

            However, this behavior can be overridden by
            snappy.number.use_field_conversion which replaces
            number_to_native_number.
            """
    return n