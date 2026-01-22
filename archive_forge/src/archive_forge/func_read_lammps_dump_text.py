import gzip
import struct
from collections import deque
from os.path import splitext
import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import paropen
from ase.quaternions import Quaternions
def read_lammps_dump_text(fileobj, index=-1, **kwargs):
    """Process cleartext lammps dumpfiles

    :param fileobj: filestream providing the trajectory data
    :param index: integer or slice object (default: get the last timestep)
    :returns: list of Atoms objects
    :rtype: list
    """
    lines = deque(fileobj.readlines())
    index_end = get_max_index(index)
    n_atoms = 0
    images = []
    cell, celldisp, pbc = (None, None, False)
    while len(lines) > n_atoms:
        line = lines.popleft()
        if 'ITEM: TIMESTEP' in line:
            n_atoms = 0
            line = lines.popleft()
        if 'ITEM: NUMBER OF ATOMS' in line:
            line = lines.popleft()
            n_atoms = int(line.split()[0])
        if 'ITEM: BOX BOUNDS' in line:
            tilt_items = line.split()[3:]
            celldatarows = [lines.popleft() for _ in range(3)]
            celldata = np.loadtxt(celldatarows)
            diagdisp = celldata[:, :2].reshape(6, 1).flatten()
            if len(celldata[0]) > 2:
                offdiag = celldata[:, 2]
                if len(tilt_items) >= 3:
                    sort_index = [tilt_items.index(i) for i in ['xy', 'xz', 'yz']]
                    offdiag = offdiag[sort_index]
            else:
                offdiag = (0.0,) * 3
            cell, celldisp = construct_cell(diagdisp, offdiag)
            if len(tilt_items) == 3:
                pbc_items = tilt_items
            elif len(tilt_items) > 3:
                pbc_items = tilt_items[3:6]
            else:
                pbc_items = ['f', 'f', 'f']
            pbc = ['p' in d.lower() for d in pbc_items]
        if 'ITEM: ATOMS' in line:
            colnames = line.split()[2:]
            datarows = [lines.popleft() for _ in range(n_atoms)]
            data = np.loadtxt(datarows, dtype=str)
            out_atoms = lammps_data_to_ase_atoms(data=data, colnames=colnames, cell=cell, celldisp=celldisp, atomsobj=Atoms, pbc=pbc, **kwargs)
            images.append(out_atoms)
        if len(images) > index_end >= 0:
            break
    return images[index]