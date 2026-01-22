import pyglet.gl as pgl
from sympy.core import S
def screen_to_model(x, y, z):
    m = get_model_matrix(c_double, pgl.glGetDoublev)
    p = get_projection_matrix(c_double, pgl.glGetDoublev)
    w = get_viewport()
    mx, my, mz = (c_double(), c_double(), c_double())
    pgl.gluUnProject(x, y, z, m, p, w, mx, my, mz)
    return (float(mx.value), float(my.value), float(mz.value))