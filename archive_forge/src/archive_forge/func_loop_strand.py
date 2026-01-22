import spherogram
def loop_strand(cs):
    """
    Get the closed loop starting at crossing strand cs
    """
    strand = [cs]
    cs = cs[0].adjacent[(cs[1] + 2) % 4]
    while cs not in strand:
        strand.append(cs)
        cs = cs[0].adjacent[(cs[1] + 2) % 4]
    return strand