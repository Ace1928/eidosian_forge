import array
import numbers
from collections.abc import Mapping, Sequence
from typing import Any, Iterable
from .const import INT_MAX_VALUE, INT_MIN_VALUE, LONG_MAX_VALUE, LONG_MIN_VALUE
from ._validate_common import ValidationError, ValidationErrorData
from .schema import extract_record_type, extract_logical_type, schema_name, parse_schema
from .logical_writers import LOGICAL_WRITERS
from ._schema_common import UnknownType
from .types import Schema, NamedSchemas
def validate_many(records: Iterable[Any], schema: Schema, raise_errors: bool=True, strict: bool=False, disable_tuple_notation: bool=False) -> bool:
    """
    Validate a list of data!

    Parameters
    ----------
    records
        List of records to validate
    schema
        Schema
    raise_errors
        If true, errors are raised for invalid data. If false, a simple
        True (valid) or False (invalid) result is returned
    strict
        If true, fields without values will raise errors rather than implicitly
        defaulting to None
    disable_tuple_notation
        If set to True, tuples will not be treated as a special case. Therefore,
        using a tuple to indicate the type of a record will not work


    Example::

        from fastavro.validation import validate_many
        schema = {...}
        records = [{...}, {...}, ...]
        validate_many(records, schema)
    """
    named_schemas: NamedSchemas = {}
    parsed_schema = parse_schema(schema, named_schemas)
    errors = []
    results = []
    for record in records:
        try:
            results.append(_validate(record, parsed_schema, named_schemas, field='', raise_errors=raise_errors, options={'strict': strict, 'disable_tuple_notation': disable_tuple_notation}))
        except ValidationError as e:
            errors.extend(e.errors)
    if raise_errors and errors:
        raise ValidationError(*errors)
    return all(results)