import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def type_draft4(validator, types, instance, schema):
    types = _utils.ensure_list(types)
    if not any((validator.is_type(instance, type) for type in types)):
        yield ValidationError(_utils.types_msg(instance, types))