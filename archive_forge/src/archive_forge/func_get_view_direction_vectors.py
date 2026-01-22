import pyglet.gl as pgl
from sympy.core import S
def get_view_direction_vectors():
    m = get_model_matrix()
    return ((m[0], m[1], m[2]), (m[4], m[5], m[6]), (m[8], m[9], m[10]))