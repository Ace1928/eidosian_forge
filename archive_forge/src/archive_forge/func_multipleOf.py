import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def multipleOf(validator, dB, instance, schema):
    if not validator.is_type(instance, 'number'):
        return
    if isinstance(dB, float):
        quotient = instance / dB
        failed = int(quotient) != quotient
    else:
        failed = instance % dB
    if failed:
        yield ValidationError('%r is not a multiple of %r' % (instance, dB))