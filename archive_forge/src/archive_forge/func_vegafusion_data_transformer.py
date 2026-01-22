from toolz import curried
import uuid
from weakref import WeakValueDictionary
from typing import (
from altair.utils._importers import import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.data import DataType, ToValuesReturnType, MaxRowsError
from altair.vegalite.data import default_data_transformer
@curried.curry
def vegafusion_data_transformer(data: DataType, max_rows: int=100000) -> Union[_ToVegaFusionReturnUrlDict, ToValuesReturnType]:
    """VegaFusion Data Transformer"""
    if hasattr(data, '__geo_interface__'):
        return default_data_transformer(data)
    elif isinstance(data, DataFrameLike):
        table_name = f'table_{uuid.uuid4()}'.replace('-', '_')
        extracted_inline_tables[table_name] = data
        return {'url': VEGAFUSION_PREFIX + table_name}
    else:
        return default_data_transformer(data)