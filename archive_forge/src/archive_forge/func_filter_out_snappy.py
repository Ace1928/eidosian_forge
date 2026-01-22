import doctest
import re
import types
def filter_out_snappy(pieces):
    ans = []
    for piece in pieces:
        if _have_snappy or not isinstance(piece, doctest.Example):
            ans.append(piece)
        elif not piece.options.get(SNAPPY_FLAG, False):
            ans.append(piece)
    return ans