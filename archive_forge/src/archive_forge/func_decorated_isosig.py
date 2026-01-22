import re
import string
def decorated_isosig(manifold, triangulation_class, ignore_cusp_ordering=False, ignore_curve_orientations=False, ignore_orientation=True):
    isosig = manifold.triangulation_isosig(decorated=False, ignore_orientation=ignore_orientation)
    if manifold.num_cusps() == 0:
        return isosig
    N = triangulation_class(isosig, remove_finite_vertices=False)
    N.set_peripheral_curves('combinatorial')
    trivial_perm = list(range(manifold.num_cusps()))
    min_encoded = None
    min_perm = None
    min_flips = None
    for tri_iso in manifold.isomorphisms_to(N):
        if manifold.is_orientable() and (not ignore_orientation) and (det(tri_iso.cusp_maps()[0]) < 0):
            continue
        perm = inverse_perm(tri_iso.cusp_images())
        if ignore_cusp_ordering:
            matrices = [tri_iso.cusp_maps()[i] for i in perm]
        else:
            matrices = tri_iso.cusp_maps()
        if ignore_curve_orientations:
            flips = determine_flips(matrices, manifold.is_orientable())
        else:
            flips = [(1, 1) for matrix in matrices]
        decorations = pack_matrices_applying_flips(matrices, flips)
        if perm == trivial_perm or ignore_cusp_ordering:
            encoded = encode_integer_list(decorations)
        else:
            encoded = encode_integer_list(perm + decorations)
        if min_encoded is None or encoded < min_encoded:
            min_encoded = encoded
            min_perm = perm
            min_flips = flips
    ans = isosig + separator + min_encoded
    if False in manifold.cusp_info('complete?'):
        if ignore_cusp_ordering:
            slopes = [manifold.cusp_info('filling')[i] for i in min_perm]
        else:
            slopes = manifold.cusp_info('filling')
        for flip, slope in zip(min_flips, slopes):
            ans += '(%g,%g)' % (supress_minus_zero(flip[0] * slope[0]), supress_minus_zero(flip[1] * slope[1]))
    return ans