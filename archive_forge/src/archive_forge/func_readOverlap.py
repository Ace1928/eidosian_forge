import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f):
    myOLP = []
    myOLP.append([])
    for ct_AN in range(1, atomnum + 1):
        myOLP.append([])
        TNO1 = Total_NumOrbs[ct_AN]
        for h_AN in range(FNAN[ct_AN] + 1):
            myOLP[ct_AN].append([])
            Gh_AN = natn[ct_AN][h_AN]
            TNO2 = Total_NumOrbs[Gh_AN]
            for i in range(TNO1):
                myOLP[ct_AN][h_AN].append(floa(f.read(8 * TNO2)))
    return myOLP