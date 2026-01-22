import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def not_draft4(validator, not_schema, instance, schema):
    if validator.is_valid(instance, not_schema):
        yield ValidationError('%r is not allowed for %r' % (not_schema, instance))