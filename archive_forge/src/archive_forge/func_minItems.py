import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def minItems(validator, mI, instance, schema):
    if validator.is_type(instance, 'array') and len(instance) < mI:
        yield ValidationError('%r is too short' % (instance,))