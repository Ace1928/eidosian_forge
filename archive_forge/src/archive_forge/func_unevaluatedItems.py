from fractions import Fraction
import re
from jsonschema._utils import (
from jsonschema.exceptions import FormatError, ValidationError
def unevaluatedItems(validator, unevaluatedItems, instance, schema):
    if not validator.is_type(instance, 'array'):
        return
    evaluated_item_indexes = find_evaluated_item_indexes_by_schema(validator, instance, schema)
    unevaluated_items = [item for index, item in enumerate(instance) if index not in evaluated_item_indexes]
    if unevaluated_items:
        error = 'Unevaluated items are not allowed (%s %s unexpected)'
        yield ValidationError(error % extras_msg(unevaluated_items))