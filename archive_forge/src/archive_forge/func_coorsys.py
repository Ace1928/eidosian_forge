import numpy as np
from ase import Atoms
from ase.units import Bohr, Ry
from ase.utils import reader, writer
def coorsys(latconst):
    a = latconst[0]
    b = latconst[1]
    c = latconst[2]
    cal = np.cos(latconst[3] * np.pi / 180.0)
    cbe = np.cos(latconst[4] * np.pi / 180.0)
    cga = np.cos(latconst[5] * np.pi / 180.0)
    sga = np.sin(latconst[5] * np.pi / 180.0)
    return np.array([[a, b * cga, c * cbe], [0, b * sga, c * (cal - cbe * cga) / sga], [0, 0, c * np.sqrt(1 - cal ** 2 - cbe ** 2 - cga ** 2 + 2 * cal * cbe * cga) / sga]]).transpose()