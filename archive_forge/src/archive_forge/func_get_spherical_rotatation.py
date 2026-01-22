import pyglet.gl as pgl
from math import sqrt as _sqrt, acos as _acos
def get_spherical_rotatation(p1, p2, width, height, theta_multiplier):
    v1 = get_sphere_mapping(p1[0], p1[1], width, height)
    v2 = get_sphere_mapping(p2[0], p2[1], width, height)
    d = min(max([dot(v1, v2), -1]), 1)
    if abs(d - 1.0) < 1e-06:
        return None
    raxis = norm(cross(v1, v2))
    rtheta = theta_multiplier * rad2deg * _acos(d)
    pgl.glPushMatrix()
    pgl.glLoadIdentity()
    pgl.glRotatef(rtheta, *raxis)
    mat = (c_float * 16)()
    pgl.glGetFloatv(pgl.GL_MODELVIEW_MATRIX, mat)
    pgl.glPopMatrix()
    return mat