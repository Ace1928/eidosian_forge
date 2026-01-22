from math import pi, sqrt
import numpy as np
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.parallel import world
from ase.utils.cext import cextension
def lti_dos1(e, w, energies, dos):
    i = e.argsort()
    e0, e1, e2, e3 = en = e[i]
    w = w[i]
    zero = energies[0]
    if len(energies) > 1:
        de = energies[1] - zero
        nn = (np.floor((en - zero) / de).astype(int) + 1).clip(0, len(energies))
    else:
        nn = (en > zero).astype(int)
    n0, n1, n2, n3 = nn
    if n1 > n0:
        s = slice(n0, n1)
        x = energies[s] - e0
        f10 = x / (e1 - e0)
        f20 = x / (e2 - e0)
        f30 = x / (e3 - e0)
        f01 = 1 - f10
        f02 = 1 - f20
        f03 = 1 - f30
        g = f20 * f30 / (e1 - e0)
        dos[:, s] += w.T.dot([f01 + f02 + f03, f10, f20, f30]) * g
    if n2 > n1:
        delta = e3 - e0
        s = slice(n1, n2)
        x = energies[s]
        f20 = (x - e0) / (e2 - e0)
        f30 = (x - e0) / (e3 - e0)
        f21 = (x - e1) / (e2 - e1)
        f31 = (x - e1) / (e3 - e1)
        f02 = 1 - f20
        f03 = 1 - f30
        f12 = 1 - f21
        f13 = 1 - f31
        g = 3 / delta * (f12 * f20 + f21 * f13)
        dos[:, s] += w.T.dot([g * f03 / 3 + f02 * f20 * f12 / delta, g * f12 / 3 + f13 * f13 * f21 / delta, g * f21 / 3 + f20 * f20 * f12 / delta, g * f30 / 3 + f31 * f13 * f21 / delta])
    if n3 > n2:
        s = slice(n2, n3)
        x = energies[s] - e3
        f03 = x / (e0 - e3)
        f13 = x / (e1 - e3)
        f23 = x / (e2 - e3)
        f30 = 1 - f03
        f31 = 1 - f13
        f32 = 1 - f23
        g = f03 * f13 / (e3 - e2)
        dos[:, s] += w.T.dot([f03, f13, f23, f30 + f31 + f32]) * g