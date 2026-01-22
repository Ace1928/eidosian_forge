import numpy as np
from ase.units import Bohr
from ase.data import atomic_numbers
def attach_charges(atoms, fileobj='ACF.dat', displacement=0.0001):
    """Attach the charges from the fileobj to the Atoms."""
    if isinstance(fileobj, str):
        with open(fileobj) as fd:
            lines = fd.readlines()
    else:
        lines = fileobj
    sep = '---------------'
    i = 0
    k = 0
    assume6columns = False
    for line in lines:
        if line[0] == '\n':
            i -= 1
        if i == 0:
            headings = line
            if 'BADER' in headings.split():
                j = headings.split().index('BADER')
            elif 'CHARGE' in headings.split():
                j = headings.split().index('CHARGE')
            else:
                print('Can\'t find keyword "BADER" or "CHARGE". Assuming the ACF.dat file has 6 columns.')
                j = 4
                assume6columns = True
        if sep in line:
            if k == 1:
                break
            k += 1
        if not i > 1:
            pass
        else:
            words = line.split()
            if assume6columns is True:
                if len(words) != 6:
                    raise IOError('Number of columns in ACF file incorrect!\nCheck that Bader program version >= 0.25')
            atom = atoms[int(words[0]) - 1]
            atom.charge = atomic_numbers[atom.symbol] - float(words[j])
            if displacement is not None:
                xyz = np.array([float(w) for w in words[1:4]])
                norm1 = np.linalg.norm(atom.position - xyz)
                norm2 = np.linalg.norm(atom.position - xyz * Bohr)
                assert norm1 < displacement or norm2 < displacement
        i += 1