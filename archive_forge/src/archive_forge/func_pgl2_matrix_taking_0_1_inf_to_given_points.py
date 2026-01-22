from snappy.SnapPy import matrix
from ..upper_halfspace.ideal_point import Infinity
def pgl2_matrix_taking_0_1_inf_to_given_points(z0, z1, zinf):
    if z0 == Infinity:
        CF = z1.parent()
        m = zinf - z1
        return matrix([[-zinf, m], [-1, 0]], ring=CF)
    if z1 == Infinity:
        CF = zinf.parent()
        return matrix([[-zinf, z0], [-1, 1]], ring=CF)
    if zinf == Infinity:
        CF = z0.parent()
        l = z0 - z1
        return matrix([[-l, z0], [0, 1]], ring=CF)
    l = z0 - z1
    m = zinf - z1
    return matrix([[-l * zinf, m * z0], [-l, m]])