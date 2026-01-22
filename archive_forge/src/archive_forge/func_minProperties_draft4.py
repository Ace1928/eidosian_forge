import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def minProperties_draft4(validator, mP, instance, schema):
    if validator.is_type(instance, 'object') and len(instance) < mP:
        yield ValidationError('%r does not have enough properties' % (instance,))