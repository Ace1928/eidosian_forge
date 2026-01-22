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
def read_castep_geom(fd, index=None, units=units_CODATA2002):
    """Reads a .geom file produced by the CASTEP GeometryOptimization task and
    returns an atoms  object.
    The information about total free energy and forces of each atom for every
    relaxation step will be stored for further analysis especially in a
    single-point calculator.
    Note that everything in the .geom file is in atomic units, which has
    been conversed to commonly used unit angstrom(length) and eV (energy).

    Note that the index argument has no effect as of now.

    Contribution by Wei-Bing Zhang. Thanks!

    Routine now accepts a filedescriptor in order to out-source the gz and
    bz2 handling to formats.py. Note that there is a fallback routine
    read_geom() that behaves like previous versions did.
    """
    from ase.calculators.singlepoint import SinglePointCalculator
    txt = fd.readlines()
    traj = []
    Hartree = units['Eh']
    Bohr = units['a0']
    for i, line in enumerate(txt):
        if line.find('<-- E') > 0:
            start_found = True
            energy = float(line.split()[0]) * Hartree
            cell = [x.split()[0:3] for x in txt[i + 1:i + 4]]
            cell = np.array([[float(col) * Bohr for col in row] for row in cell])
        if line.find('<-- R') > 0 and start_found:
            start_found = False
            geom_start = i
            for i, line in enumerate(txt[geom_start:]):
                if line.find('<-- F') > 0:
                    geom_stop = i + geom_start
                    break
            species = [line.split()[0] for line in txt[geom_start:geom_stop]]
            geom = np.array([[float(col) * Bohr for col in line.split()[2:5]] for line in txt[geom_start:geom_stop]])
            forces = np.array([[float(col) * Hartree / Bohr for col in line.split()[2:5]] for line in txt[geom_stop:geom_stop + (geom_stop - geom_start)]])
            image = ase.Atoms(species, geom, cell=cell, pbc=True)
            image.calc = SinglePointCalculator(atoms=image, energy=energy, forces=forces)
            traj.append(image)
    if index is None:
        return traj
    else:
        return traj[index]