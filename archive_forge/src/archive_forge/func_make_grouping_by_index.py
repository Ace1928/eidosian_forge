from dash.exceptions import InvalidCallbackReturnValue
from ._utils import AttributeDict, stringify_id
def make_grouping_by_index(schema, flat_values):
    """
    Make a grouping like the provided grouping schema, with scalar values drawn from a
    flat list by index.

    Note: Scalar values in schema are not used

    :param schema: Grouping value encoding the structure of the grouping to return
    :param flat_values: List of values with length matching the grouping_len of schema.
        Elements of flat_values will become the scalar values in the resulting grouping
    """

    def _perform_make_grouping_like(value, next_values):
        if isinstance(value, (tuple, list)):
            return list((_perform_make_grouping_like(el, next_values) for i, el in enumerate(value)))
        if isinstance(value, dict):
            return {k: _perform_make_grouping_like(v, next_values) for i, (k, v) in enumerate(value.items())}
        return next_values.pop(0)
    if not isinstance(flat_values, list):
        raise ValueError(f'The flat_values argument must be a list. Received value of type {type(flat_values)}')
    expected_length = len(flatten_grouping(schema))
    if len(flat_values) != expected_length:
        raise ValueError(f'The specified grouping pattern requires {expected_length} elements but received {len(flat_values)}\n    Grouping pattern: {repr(schema)}\n    Values: {flat_values}')
    return _perform_make_grouping_like(schema, list(flat_values))