from collections import OrderedDict
from datetime import date, time
import numpy as np
import pandas as pd
import pyarrow as pa
def dataframe_with_lists(include_index=False, parquet_compatible=False):
    """
    Dataframe with list columns of every possible primitive type.

    Returns
    -------
    df: pandas.DataFrame
    schema: pyarrow.Schema
        Arrow schema definition that is in line with the constructed df.
    parquet_compatible: bool
        Exclude types not supported by parquet
    """
    arrays = OrderedDict()
    fields = []
    fields.append(pa.field('int64', pa.list_(pa.int64())))
    arrays['int64'] = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4], None, [], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2, dtype=np.int64)[::2]]
    fields.append(pa.field('double', pa.list_(pa.float64())))
    arrays['double'] = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [0.0, 1.0, 2.0, 3.0, 4.0], None, [], np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] * 2)[::2]]
    fields.append(pa.field('bytes_list', pa.list_(pa.binary())))
    arrays['bytes_list'] = [[b'1', b'f'], None, [b'1'], [b'1', b'2', b'3'], []]
    fields.append(pa.field('str_list', pa.list_(pa.string())))
    arrays['str_list'] = [['1', 'ä'], None, ['1'], ['1', '2', '3'], []]
    date_data = [[], [date(2018, 1, 1), date(2032, 12, 30)], [date(2000, 6, 7)], None, [date(1969, 6, 9), date(1972, 7, 3)]]
    time_data = [[time(23, 11, 11), time(1, 2, 3), time(23, 59, 59)], [], [time(22, 5, 59)], None, [time(0, 0, 0), time(18, 0, 2), time(12, 7, 3)]]
    temporal_pairs = [(pa.date32(), date_data), (pa.date64(), date_data), (pa.time32('s'), time_data), (pa.time32('ms'), time_data), (pa.time64('us'), time_data)]
    if not parquet_compatible:
        temporal_pairs += [(pa.time64('ns'), time_data)]
    for value_type, data in temporal_pairs:
        field_name = '{}_list'.format(value_type)
        field_type = pa.list_(value_type)
        field = pa.field(field_name, field_type)
        fields.append(field)
        arrays[field_name] = data
    if include_index:
        fields.append(pa.field('__index_level_0__', pa.int64()))
    df = pd.DataFrame(arrays)
    schema = pa.schema(fields)
    return (df, schema)