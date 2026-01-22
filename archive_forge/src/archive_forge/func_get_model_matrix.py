import pyglet.gl as pgl
from sympy.core import S
def get_model_matrix(array_type=c_float, glGetMethod=pgl.glGetFloatv):
    """
    Returns the current modelview matrix.
    """
    m = (array_type * 16)()
    glGetMethod(pgl.GL_MODELVIEW_MATRIX, m)
    return m