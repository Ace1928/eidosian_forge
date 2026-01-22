import abc
from typing import TYPE_CHECKING, Dict, List, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.dtypes.common import is_string_dtype
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import EMPTY_ARROW_TABLE, ColNameCodec, get_common_arrow_type
from .db_worker import DbTable
from .expr import InputRefExpr, LiteralExpr, OpExpr
def execute_arrow(self, tables: Union[pa.Table, List[pa.Table]]) -> pa.Table:
    """
        Concat frames' rows using Arrow API.

        Parameters
        ----------
        tables : pa.Table or list of pa.Table

        Returns
        -------
        pyarrow.Table
            The resulting table.
        """
    if len(self.columns) == 0:
        frames = self.input
        if len(frames) == 0:
            return EMPTY_ARROW_TABLE
        elif self.ignore_index:
            idx = pandas.RangeIndex(0, sum((len(frame.index) for frame in frames)))
        else:
            idx = frames[0].index.append([f.index for f in frames[1:]])
        idx_cols = ColNameCodec.mangle_index_names(idx.names)
        idx_df = pandas.DataFrame(index=idx).reset_index()
        obj_cols = idx_df.select_dtypes(include=['object']).columns.tolist()
        if len(obj_cols) != 0:
            idx_df[obj_cols] = idx_df[obj_cols].astype(str)
        idx_table = pa.Table.from_pandas(idx_df, preserve_index=False)
        return idx_table.rename_columns(idx_cols)
    if isinstance(tables, pa.Table):
        assert len(self.input) == 1
        return tables
    try:
        return pa.concat_tables(tables)
    except pa.lib.ArrowInvalid:
        fields: Dict[str, pa.Field] = {}
        for table in tables:
            for col_name in table.column_names:
                field = table.field(col_name)
                cur_field = fields.get(col_name, None)
                if cur_field is None or cur_field.type != get_common_arrow_type(cur_field.type, field.type):
                    fields[col_name] = field
        schema = pa.schema(list(fields.values()))
        for i, table in enumerate(tables):
            tables[i] = pa.table(table.columns, schema=schema)
        return pa.concat_tables(tables)