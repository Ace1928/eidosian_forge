from math import sqrt
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.gui.defaults import read_defaults
from ase.io import read, write, string2index
from ase.gui.i18n import _
from ase.geometry import find_mic
import warnings
def dih(n1, n2, n3, n4):
    a = R[n2] - R[n1]
    b = R[n3] - R[n2]
    c = R[n4] - R[n3]
    bxa = np.cross(b, a)
    bxa /= np.sqrt(np.vdot(bxa, bxa))
    cxb = np.cross(c, b)
    cxb /= np.sqrt(np.vdot(cxb, cxb))
    angle = np.vdot(bxa, cxb)
    if angle < -1:
        angle = -1
    if angle > 1:
        angle = 1
    angle = np.arccos(angle)
    if np.vdot(bxa, c) > 0:
        angle = 2 * np.pi - angle
    return angle * 180.0 / np.pi