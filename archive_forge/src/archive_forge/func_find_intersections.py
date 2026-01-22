from reportlab.graphics.shapes import Drawing, Polygon, Line
from math import pi
def find_intersections(data, small=0):
    """
    data is a sequence of series
    each series is a list of (x,y) coordinates
    where x & y are ints or floats

    find_intersections returns a sequence of 4-tuples
        i, j, x, y

    where i is a data index j is an insertion position for data[i]
    and x, y are coordinates of an intersection of series data[i]
    with some other series. If correctly implemented we get all such
    intersections. We don't count endpoint intersections and consider
    parallel lines as non intersecting (even when coincident).
    We ignore segments that have an estimated size less than small.
    """
    S = []
    a = S.append
    for s in range(len(data)):
        ds = data[s]
        if not ds:
            continue
        n = len(ds)
        if n == 1:
            continue
        for i in range(1, n):
            seg = _Segment(s, i, data)
            if seg.a + abs(seg.b) >= small:
                a(seg)
    S.sort(key=_segKey)
    I = []
    n = len(S)
    for i in range(0, n - 1):
        s = S[i]
        for j in range(i + 1, n):
            if s.intersect(S[j], I) == 1:
                break
    I.sort()
    return I