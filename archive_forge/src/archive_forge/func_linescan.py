import numpy as np
from ase.io.jsonio import read_json, write_json
def linescan(self, bias, current, p1, p2, npoints=50, z0=None):
    """Constant current line scan.

        Example::

            stm = STM(...)
            z = ...  # tip position
            c = stm.get_averaged_current(-1.0, z)
            stm.linescan(-1.0, c, (1.2, 0.0), (1.2, 3.0))
        """
    heights = self.scan(bias, current, z0)[2]
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    d = p2 - p1
    s = np.dot(d, d) ** 0.5
    cell = self.cell[:2, :2]
    shape = np.array(heights.shape, float)
    M = np.linalg.inv(cell)
    line = np.empty(npoints)
    for i in range(npoints):
        p = p1 + i * d / (npoints - 1)
        q = np.dot(p, M) * shape
        line[i] = interpolate(q, heights)
    return (np.linspace(0, s, npoints), line)