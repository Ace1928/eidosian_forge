from .sage_helper import _within_sage
from .pari import *
import re
def use_field_conversion(func):
    global number_to_native_number
    if func == 'sage':

        def number_to_native_number(n):
            """
            Converts a SnapPy number to the corresponding SageMath type.

            In general snappy.number.number_to_native_number converts a SnapPy number to
            the corresponding SageMath type (when in SageMath) or just returns
            the SnapPy number itself (when SageMath is not available).

            However, this behavior can be overridden by
            snappy.number.use_field_conversion which replaces
            number_to_native_number.
            """
            return n.sage()
    elif func == 'snappy':

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
    else:
        number_to_native_number = func