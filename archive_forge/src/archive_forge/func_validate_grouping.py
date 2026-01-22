from dash.exceptions import InvalidCallbackReturnValue
from ._utils import AttributeDict, stringify_id
def validate_grouping(grouping, schema, full_schema=None, path=()):
    """
    Validate that the provided grouping conforms to the provided schema.
    If not, raise a SchemaValidationError
    """
    if full_schema is None:
        full_schema = schema
    if isinstance(schema, (tuple, list)):
        SchemaTypeValidationError.check(grouping, full_schema, path, (tuple, list))
        SchemaLengthValidationError.check(grouping, full_schema, path, len(schema))
        for i, (g, s) in enumerate(zip(grouping, schema)):
            validate_grouping(g, s, full_schema=full_schema, path=path + (i,))
    elif isinstance(schema, dict):
        SchemaTypeValidationError.check(grouping, full_schema, path, dict)
        SchemaKeysValidationError.check(grouping, full_schema, path, set(schema))
        for k in schema:
            validate_grouping(grouping[k], schema[k], full_schema=full_schema, path=path + (k,))
    else:
        pass