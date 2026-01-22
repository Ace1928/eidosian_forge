import json
import os
from .mcomplex_with_memory import McomplexWithMemory
from .mcomplex_with_expansion import McomplexWithExpansion
from .mcomplex_with_link import (link_triangulation,
def good_simplification(manifold, max_tries=5):
    """
    >>> M = Manifold('K4a1')
    >>> ans = good_simplification(M, max_tries=1)
    >>> len(ans[1]) > ans[2]
    True
    """
    tris_with_moves = simplifications(manifold, max_tries)
    if tris_with_moves:
        M, moves, unexpanded = tris_with_moves[0]
        T = McomplexWithExpansion(M._triangulation_data())
        T.perform_moves(moves, tet_stop_num=5)
        final_moves = geodesic_moves(T)
        T.perform_moves(final_moves)
        return (M, T.move_memory, unexpanded)