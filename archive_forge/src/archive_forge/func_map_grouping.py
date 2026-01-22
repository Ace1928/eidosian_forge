from dash.exceptions import InvalidCallbackReturnValue
from ._utils import AttributeDict, stringify_id
def map_grouping(fn, grouping):
    """
    Map a function over all of the scalar values of a grouping, maintaining the
    grouping structure

    :param fn: Single-argument function that accepts and returns scalar grouping values
    :param grouping: The grouping to map the function over
    :return: A new grouping with the same structure as input grouping with scalar
        values updated by the input function.
    """
    if isinstance(grouping, (tuple, list)):
        return [map_grouping(fn, g) for g in grouping]
    if isinstance(grouping, dict):
        return AttributeDict({k: map_grouping(fn, g) for k, g in grouping.items()})
    return fn(grouping)