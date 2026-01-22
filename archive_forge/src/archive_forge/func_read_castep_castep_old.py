import os
import re
import warnings
import numpy as np
from copy import deepcopy
import ase
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian
from ase.utils import atoms_to_spglib_cell
import ase.units
def read_castep_castep_old(fd, index=None):
    """
    DEPRECATED
    Now replaced by ase.calculators.castep.Castep.read(). Left in for future
    reference and backwards compatibility needs, as well as a fallback for
    when castep_keywords.py can't be created.

    Reads a .castep file and returns an atoms  object.
    The calculator information will be stored in the calc attribute.
    If more than one SCF step is found, a list of all steps
    will be stored in the traj attribute.

    Note that the index argument has no effect as of now.

    Please note that this routine will return an atom ordering as found
    within the castep file. This means that the species will be ordered by
    ascending atomic numbers. The atoms witin a species are ordered as given
    in the original cell file.
    """
    from ase.calculators.singlepoint import SinglePointCalculator
    lines = fd.readlines()
    traj = []
    energy_total = None
    energy_0K = None
    for i, line in enumerate(lines):
        if 'NB est. 0K energy' in line:
            energy_0K = float(line.split()[6])
        elif 'NB dispersion corrected est. 0K energy*' in line:
            energy_0K = float(line.split()[-2])
        elif 'Final energy, E' in line:
            energy_total = float(line.split()[4])
        elif 'Dispersion corrected final energy' in line:
            pass
        elif 'Dispersion corrected final free energy' in line:
            pass
        elif 'dispersion corrected est. 0K energy' in line:
            pass
        elif 'Unit Cell' in line:
            cell = [x.split()[0:3] for x in lines[i + 3:i + 6]]
            cell = np.array([[float(col) for col in row] for row in cell])
        elif 'Cell Contents' in line:
            geom_starts = i
            start_found = False
            for j, jline in enumerate(lines[geom_starts:]):
                if jline.find('xxxxx') > 0 and start_found:
                    geom_stop = j + geom_starts
                    break
                if jline.find('xxxx') > 0 and (not start_found):
                    geom_start = j + geom_starts + 4
                    start_found = True
            species = [line.split()[1] for line in lines[geom_start:geom_stop]]
            geom = np.dot(np.array([[float(col) for col in line.split()[3:6]] for line in lines[geom_start:geom_stop]]), cell)
        elif 'Writing model to' in line:
            atoms = ase.Atoms(cell=cell, pbc=True, positions=geom, symbols=''.join(species))
            if energy_0K:
                energy = energy_0K
            else:
                energy = energy_total
            sp_calc = SinglePointCalculator(atoms=atoms, energy=energy, forces=None, magmoms=None, stress=None)
            atoms.calc = sp_calc
            traj.append(atoms)
    if index is None:
        return traj
    else:
        return traj[index]