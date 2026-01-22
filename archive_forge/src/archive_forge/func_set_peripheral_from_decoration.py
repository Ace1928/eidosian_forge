import re
import string
def set_peripheral_from_decoration(manifold, decoration):
    """
    The manifold is assumed to already have a triangulation created
    from the "bare" isosig.
    """
    dec = decode_integer_list(decoration)
    manifold.set_peripheral_curves('combinatorial')
    n = manifold.num_cusps()
    if len(dec) == 4 * n:
        cobs = as_two_by_two_matrices(dec)
    else:
        assert len(dec) == 5 * n
        manifold._reindex_cusps(dec[:n])
        cobs = as_two_by_two_matrices(dec[n:])
    if det(cobs[0]) < 0 and manifold.is_orientable():
        manifold.reverse_orientation()
        cobs = [[(-a, b), (-c, d)] for [(a, b), (c, d)] in cobs]
    manifold.set_peripheral_curves(cobs)