from __future__ import annotations
from typing import TYPE_CHECKING
from pandas._libs import lib
from pandas.core.dtypes.missing import notna
from pandas.core.algorithms import factorize
from pandas.core.indexes.api import MultiIndex
from pandas.core.series import Series
def sparse_series_to_coo(ss: Series, row_levels: Iterable[int]=(0,), column_levels: Iterable[int]=(1,), sort_labels: bool=False) -> tuple[scipy.sparse.coo_matrix, list[IndexLabel], list[IndexLabel]]:
    """
    Convert a sparse Series to a scipy.sparse.coo_matrix using index
    levels row_levels, column_levels as the row and column
    labels respectively. Returns the sparse_matrix, row and column labels.
    """
    import scipy.sparse
    if ss.index.nlevels < 2:
        raise ValueError('to_coo requires MultiIndex with nlevels >= 2.')
    if not ss.index.is_unique:
        raise ValueError('Duplicate index entries are not allowed in to_coo transformation.')
    row_levels = [ss.index._get_level_number(x) for x in row_levels]
    column_levels = [ss.index._get_level_number(x) for x in column_levels]
    v, i, j, rows, columns = _to_ijv(ss, row_levels=row_levels, column_levels=column_levels, sort_labels=sort_labels)
    sparse_matrix = scipy.sparse.coo_matrix((v, (i, j)), shape=(len(rows), len(columns)))
    return (sparse_matrix, rows, columns)