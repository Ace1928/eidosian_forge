import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def required_draft4(validator, required, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    for property in required:
        if property not in instance:
            yield ValidationError('%r is a required property' % property)