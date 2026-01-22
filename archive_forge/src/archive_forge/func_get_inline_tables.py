from toolz import curried
import uuid
from weakref import WeakValueDictionary
from typing import (
from altair.utils._importers import import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.data import DataType, ToValuesReturnType, MaxRowsError
from altair.vegalite.data import default_data_transformer
def get_inline_tables(vega_spec: dict) -> Dict[str, DataFrameLike]:
    """Get the inline tables referenced by a Vega specification

    Note: This function should only be called on a Vega spec that corresponds
    to a chart that was processed by the vegafusion_data_transformer.
    Furthermore, this function may only be called once per spec because
    the returned dataframes are deleted from internal storage.

    Parameters
    ----------
    vega_spec: dict
        A Vega specification dict

    Returns
    -------
    dict from str to dataframe
        dict from inline dataset name to dataframe object
    """
    table_names = get_inline_table_names(vega_spec)
    tables = {}
    for table_name in table_names:
        try:
            tables[table_name] = extracted_inline_tables.pop(table_name)
        except KeyError:
            pass
    return tables