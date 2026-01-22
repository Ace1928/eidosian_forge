import spherogram
def cross_strand(tangle, i):
    """
    Give a list of the crossing strands encountered starting at
    a strand on the boundary of a tangle and moving to the other
    end of that strand.
    """
    if i >= tangle.boundary[0] + tangle.boundary[1]:
        raise ValueError('not a valid start position for strand')
    cs = tangle.adjacent[i]
    strand = [cs]
    while (cs[0], (cs[1] + 2) % 4) not in tangle.adjacent:
        cs = cs[0].adjacent[(cs[1] + 2) % 4]
        strand.append(cs)
    return strand