import numpy as np
from ase.io.fortranfile import FortranFile
def read_rho(fname):
    """Read unformatted Siesta charge density file"""
    fh = FortranFile(fname)
    x = fh.readReals('d')
    if len(x) != 3 * 3:
        raise IOError('Failed to read cell vectors')
    x = fh.readInts()
    if len(x) != 4:
        raise IOError('Failed to read grid size')
    gpts = x
    rho = np.zeros(gpts)
    for ispin in range(gpts[3]):
        for n3 in range(gpts[2]):
            for n2 in range(gpts[1]):
                x = fh.readReals('f')
                if len(x) != gpts[0]:
                    raise IOError('Failed to read RHO[:,%i,%i,%i]' % (n2, n3, ispin))
                rho[:, n2, n3, ispin] = x
    fh.close()
    return rho