from math import pi, sqrt
from ipywidgets import register
from .._base.Three import ThreeWidget
from .Object3D_autogen import Object3D as Object3DBase
def setRotationFromMatrix(self, m):
    """
        m is a 3 by 3 matrix, as a list of rows.  The columns of this matrix are
        the vectors x, y, and z
        """
    x = m[0:3]
    y = m[3:6]
    z = m[6:9]
    trace = x[0] + y[1] + z[2]
    if trace > 0:
        s = 0.5 / sqrt(trace + 1)
        self.quaternion = [(y[2] - z[1]) * s, (z[0] - x[2]) * s, (x[1] - y[0]) * s, 0.25 / s]
    elif x[0] > y[1] and x[0] > z[2]:
        s = 2.0 * sqrt(1.0 + x[0] - y[1] - z[2])
        self.quaternion = [0.25 * s, (y[0] + x[1]) / s, (z[0] + x[2]) / s, (y[2] - z[1]) / s]
    elif y[1] > z[2]:
        s = 2.0 * sqrt(1.0 + y[1] - x[0] - z[2])
        self.quaternion = [(y[0] + x[1]) / s, 0.25 * s, (z[1] + y[2]) / s, (z[0] - x[2]) / s]
    else:
        s = 2.0 * sqrt(1.0 + z[2] - x[0] - y[1])
        self.quaternion = [(z[0] + x[2]) / s, (z[1] + y[2]) / s, 0.25 * s, (x[1] - y[0]) / s]