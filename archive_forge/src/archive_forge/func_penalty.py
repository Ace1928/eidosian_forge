from . import matrix
def penalty(edge):
    if isinstance(edge, ShortEdge):
        return penalties[0]
    if isinstance(edge, MiddleEdge):
        return penalties[1]
    return penalties[2]