import struct
from io import BytesIO
from functools import wraps
import warnings
def raise_conversion_error(function):
    """ Wrap any raised struct.errors in a ConversionError. """

    @wraps(function)
    def result(self, value):
        try:
            return function(self, value)
        except struct.error as e:
            raise ConversionError(e.args[0]) from None
    return result