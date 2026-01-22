import spherogram
def min_isosig(tangle, root=None, over_or_under=False):
    if root is not None:
        cs_name = cslabel(root)
    isosigs = []
    for i in range(tangle.boundary[0] + tangle.boundary[1]):
        rotated_tangle = tangle.reshape((tangle.boundary[0] + tangle.boundary[1], 0), i)
        if root is not None:
            rotated_root = crossing_strand_from_name(rotated_tangle, cs_name)
        else:
            rotated_root = None
        isosigs.append(isosig(rotated_tangle, root=rotated_root, over_or_under=over_or_under))
    return min(isosigs)