import math
from typing import Optional, Sequence
import numpy as np
from ase.atoms import Atoms
import ase.data
def make_unit_cell(self):
    """Make the unit cell."""
    self.natoms = self.calc_num_atoms()
    self.nput = 0
    self.atoms = np.zeros((self.natoms, 3), float)
    self.elements = np.zeros(self.natoms, int)
    self.farpoint = sum(self.directions)
    sqrad = 0
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                vect = i * self.directions[0] + j * self.directions[1] + k * self.directions[2]
                if np.dot(vect, vect) > sqrad:
                    sqrad = np.dot(vect, vect)
    del i, j, k
    for istart, istep in ((0, 1), (-1, -1)):
        i = istart
        icont = True
        while icont:
            nj = 0
            for jstart, jstep in ((0, 1), (-1, -1)):
                j = jstart
                jcont = True
                while jcont:
                    nk = 0
                    for kstart, kstep in ((0, 1), (-1, -1)):
                        k = kstart
                        kcont = True
                        while kcont:
                            point = np.array((i, j, k))
                            if self.inside(point):
                                self.put_atom(point)
                                nk += 1
                                nj += 1
                            if np.dot(point, point) > sqrad:
                                assert not self.inside(point)
                                kcont = False
                            k += kstep
                    if i * i + j * j > sqrad:
                        jcont = False
                    j += jstep
            if i * i > sqrad:
                icont = False
            i += istep
    assert self.nput == self.natoms