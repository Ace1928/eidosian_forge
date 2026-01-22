import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def readHam(SpinP_switch, FNAN, atomnum, Total_NumOrbs, natn, f):
    Hks = []
    for spin in range(SpinP_switch + 1):
        Hks.append([])
        Hks[spin].append([np.zeros(FNAN[0] + 1)])
        for ct_AN in range(1, atomnum + 1):
            Hks[spin].append([])
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN] + 1):
                Hks[spin][ct_AN].append([])
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                for i in range(TNO1):
                    Hks[spin][ct_AN][h_AN].append(floa(f.read(8 * TNO2)))
    return Hks