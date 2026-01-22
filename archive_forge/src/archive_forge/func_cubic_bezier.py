import tkinter
from tkinter.constants import *
def cubic_bezier(document, coords):
    """cubic bezier polyline"""
    element = document.createElement('path')
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    path = ['M%s %s' % points[0]]
    for n in range(1, len(points), 3):
        A, B, C = points[n:n + 3]
        path.append('C%s,%s %s,%s %s,%s' % (A[0], A[1], B[0], B[1], C[0], C[1]))
    element.setAttribute('d', ' '.join(path))
    return element