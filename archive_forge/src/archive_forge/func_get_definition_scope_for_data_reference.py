from typing import List, Optional, Tuple, Dict, Iterable, overload, Union
from altair import (
from altair.utils._vegafusion_data import get_inline_tables, import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.schemapi import Undefined
def get_definition_scope_for_data_reference(vega_spec: dict, data_name: str, usage_scope: Scope) -> Optional[Scope]:
    """Return the scope that a dataset is defined at, for a given usage scope

    Parameters
    ----------
    vega_spec: dict
        Top-level Vega specification
    data_name: str
        The name of a dataset reference
    usage_scope: tuple of int
        The scope that the dataset is referenced in

    Returns
    -------
    tuple of int
        The scope where the referenced dataset is defined,
        or None if no such dataset is found

    Examples
    --------
    >>> spec = {
    ...     "data": [
    ...         {"name": "data1"}
    ...     ],
    ...     "marks": [
    ...         {
    ...             "type": "group",
    ...             "data": [
    ...                 {"name": "data2"}
    ...             ],
    ...             "marks": [{
    ...                 "type": "symbol",
    ...                 "encode": {
    ...                     "update": {
    ...                         "x": {"field": "x", "data": "data1"},
    ...                         "y": {"field": "y", "data": "data2"},
    ...                     }
    ...                 }
    ...             }]
    ...         }
    ...     ]
    ... }

    data1 is referenced at scope [0] and defined at scope []
    >>> get_definition_scope_for_data_reference(spec, "data1", (0,))
    ()

    data2 is referenced at scope [0] and defined at scope [0]
    >>> get_definition_scope_for_data_reference(spec, "data2", (0,))
    (0,)

    If data2 is not visible at scope [] (the top level),
    because it's defined in scope [0]
    >>> repr(get_definition_scope_for_data_reference(spec, "data2", ()))
    'None'
    """
    for i in reversed(range(len(usage_scope) + 1)):
        scope = usage_scope[:i]
        datasets = get_datasets_for_scope(vega_spec, scope)
        if data_name in datasets:
            return scope
    return None