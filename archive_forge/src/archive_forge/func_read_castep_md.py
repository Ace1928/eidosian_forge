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
def read_castep_md(fd, index=None, return_scalars=False, units=units_CODATA2002):
    """Reads a .md file written by a CASTEP MolecularDynamics task
    and returns the trajectory stored therein as a list of atoms object.

    Note that the index argument has no effect as of now."""
    from ase.calculators.singlepoint import SinglePointCalculator
    factors = {'t': units['t0'] * 1000000000000000.0, 'E': units['Eh'], 'T': units['Eh'] / units['kB'], 'P': units['Eh'] / units['a0'] ** 3 * units['Pascal'], 'h': units['a0'], 'hv': units['a0'] / units['t0'], 'S': units['Eh'] / units['a0'] ** 3, 'R': units['a0'], 'V': np.sqrt(units['Eh'] / units['me']), 'F': units['Eh'] / units['a0']}
    lines = fd.readlines()
    L = 0
    while 'END header' not in lines[L]:
        L += 1
    l_end_header = L
    lines = lines[l_end_header + 1:]
    times = []
    energies = []
    temperatures = []
    pressures = []
    traj = []
    time = None
    Epot = None
    Ekin = None
    EH = None
    temperature = None
    pressure = None
    symbols = None
    positions = None
    cell = None
    velocities = None
    symbols = []
    positions = []
    velocities = []
    forces = []
    cell = np.eye(3)
    cell_velocities = []
    stress = []
    for L, line in enumerate(lines):
        fields = line.split()
        if len(fields) == 0:
            if L != 0:
                times.append(time)
                energies.append([Epot, EH, Ekin])
                temperatures.append(temperature)
                pressures.append(pressure)
                atoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell)
                atoms.set_velocities(velocities)
                if len(stress) == 0:
                    atoms.calc = SinglePointCalculator(atoms=atoms, energy=Epot, forces=forces)
                else:
                    atoms.calc = SinglePointCalculator(atoms=atoms, energy=Epot, forces=forces, stress=stress)
                traj.append(atoms)
            symbols = []
            positions = []
            velocities = []
            forces = []
            cell = []
            cell_velocities = []
            stress = []
            continue
        if len(fields) == 1:
            time = factors['t'] * float(fields[0])
            continue
        if fields[-1] == 'E':
            E = [float(x) for x in fields[0:3]]
            Epot, EH, Ekin = [factors['E'] * Ei for Ei in E]
            continue
        if fields[-1] == 'T':
            temperature = factors['T'] * float(fields[0])
            continue
        if fields[-1] == 'P':
            pressure = factors['P'] * float(fields[0])
            continue
        if fields[-1] == 'h':
            h = [float(x) for x in fields[0:3]]
            cell.append([factors['h'] * hi for hi in h])
            continue
        if fields[-1] == 'hv':
            hv = [float(x) for x in fields[0:3]]
            cell_velocities.append([factors['hv'] * hvi for hvi in hv])
            continue
        if fields[-1] == 'S':
            S = [float(x) for x in fields[0:3]]
            stress.append([factors['S'] * Si for Si in S])
            continue
        if fields[-1] == 'R':
            symbols.append(fields[0])
            R = [float(x) for x in fields[2:5]]
            positions.append([factors['R'] * Ri for Ri in R])
            continue
        if fields[-1] == 'V':
            V = [float(x) for x in fields[2:5]]
            velocities.append([factors['V'] * Vi for Vi in V])
            continue
        if fields[-1] == 'F':
            F = [float(x) for x in fields[2:5]]
            forces.append([factors['F'] * Fi for Fi in F])
            continue
    if index is None:
        pass
    else:
        traj = traj[index]
    if return_scalars:
        data = [times, energies, temperatures, pressures]
        return (data, traj)
    else:
        return traj