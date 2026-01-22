import tkinter
from tkinter.constants import *
def smoothpolygon(document, coords):
    """smoothed filled polygon"""
    element = document.createElement('path')
    path = []
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

    def pt(points):
        p = points
        n = len(points)
        for i in range(0, len(points)):
            a = p[(i - 1) % n]
            b = p[i]
            c = p[(i + 1) % n]
            yield (lerp(a, b, 0.5), b, lerp(b, c, 0.5))
    for i, (A, B, C) in enumerate(pt(points)):
        if i == 0:
            path.append('M%s,%s Q%s,%s %s,%s' % (A[0], A[1], B[0], B[1], C[0], C[1]))
        else:
            path.append('T%s,%s' % (C[0], C[1]))
    path.append('z')
    element.setAttribute('d', ' '.join(path))
    return element