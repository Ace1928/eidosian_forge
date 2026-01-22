from typing import TYPE_CHECKING, List, Union
def take_table(table: 'pyarrow.Table', indices: Union[List[int], 'pyarrow.Array', 'pyarrow.ChunkedArray']) -> 'pyarrow.Table':
    """Select rows from the table.

    This method is an alternative to pyarrow.Table.take(), which breaks for
    extension arrays. This is exposed as a static method for easier use on
    intermediate tables, not underlying an ArrowBlockAccessor.
    """
    from ray.air.util.transform_pyarrow import _concatenate_extension_column, _is_column_extension_type
    if any((_is_column_extension_type(col) for col in table.columns)):
        new_cols = []
        for col in table.columns:
            if _is_column_extension_type(col) and col.num_chunks > 1:
                col = _concatenate_extension_column(col)
            new_cols.append(col.take(indices))
        table = pyarrow.Table.from_arrays(new_cols, schema=table.schema)
    else:
        table = table.take(indices)
    return table