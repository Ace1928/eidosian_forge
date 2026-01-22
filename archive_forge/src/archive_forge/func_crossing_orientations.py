import spherogram
def crossing_orientations(strands):
    """
    Given the strands, compute the orientations (+1 or -1) for each crossing
    in the tangle
    """
    orientations = {}
    over_or_under = {}
    css_seen = []
    for strand in strands:
        for cs in strand:
            for seen_cs in css_seen:
                if cs[0] == seen_cs[0]:
                    orientation = (cs[1] - seen_cs[1]) % 4
                    if orientation == 3:
                        orientation = -1
                    orientations[cs[0]] = orientation
                    over_or_under[cs[0]] = cs[1] % 2
                    break
            css_seen.append(cs)
    return (orientations, over_or_under)