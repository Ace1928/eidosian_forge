from pyomo.common.deprecation import deprecated
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
from pyomo.common.dependencies import networkx as nx
Compute ordered partitions of the matrix's rows and columns that
    permute the matrix to block lower triangular form

    Subsets in the partition correspond to diagonal blocks in the block
    triangularization. The order is topological, with ties broken
    "lexicographically".

    Parameters
    ----------
    matrix: ``scipy.sparse.coo_matrix``
        Matrix whose rows and columns will be permuted
    matching: ``dict``
        A perfect matching. Maps rows to columns *and* columns back to rows.

    Returns
    -------
    row_partition: list of lists
        A partition of rows. The inner lists hold integer row coordinates.
    col_partition: list of lists
        A partition of columns. The inner lists hold integer column coordinates.


    .. note::

       **Breaking change in Pyomo 6.5.0**

       The pre-6.5.0 ``block_triangularize`` function returned maps from
       each row or column to the index of its block in a block
       lower triangularization as the original intent of this function
       was to identify when coordinates do or don't share a diagonal block
       in this partition. Since then, the dominant use case of
       ``block_triangularize`` has been to partition variables and
       constraints into these blocks and inspect or solve each block
       individually. A natural return type for this functionality is the
       ordered partition of rows and columns, as lists of lists.
       This functionality was previously available via the
       ``get_diagonal_blocks`` method, which was confusing as it did not
       capture that the partition was the diagonal of a block
       *triangularization* (as opposed to diagonalization). The pre-6.5.0
       functionality of ``block_triangularize`` is still available via the
       ``map_coords_to_block_triangular_indices`` function.

    