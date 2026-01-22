import re
from referencing.jsonschema import lookup_recursive_ref
from jsonschema import _utils
from jsonschema.exceptions import ValidationError
def recursiveRef(validator, recursiveRef, instance, schema):
    resolved = lookup_recursive_ref(validator._resolver)
    yield from validator.descend(instance, resolved.contents, resolver=resolved.resolver)