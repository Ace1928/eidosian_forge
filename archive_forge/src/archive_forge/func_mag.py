import pyglet.gl as pgl
from math import sqrt as _sqrt, acos as _acos
def mag(a):
    return _sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)