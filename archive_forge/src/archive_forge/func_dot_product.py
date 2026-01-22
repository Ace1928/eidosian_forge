import pyglet.gl as pgl
from sympy.core import S
def dot_product(v1, v2):
    return sum((v1[i] * v2[i] for i in range(3)))