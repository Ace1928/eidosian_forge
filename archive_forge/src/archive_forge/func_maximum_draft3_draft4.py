import re
from referencing.jsonschema import lookup_recursive_ref
from jsonschema import _utils
from jsonschema.exceptions import ValidationError
def maximum_draft3_draft4(validator, maximum, instance, schema):
    if not validator.is_type(instance, 'number'):
        return
    if schema.get('exclusiveMaximum', False):
        failed = instance >= maximum
        cmp = 'greater than or equal to'
    else:
        failed = instance > maximum
        cmp = 'greater than'
    if failed:
        message = f'{instance!r} is {cmp} the maximum of {maximum!r}'
        yield ValidationError(message)