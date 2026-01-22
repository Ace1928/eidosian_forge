import time
import warnings
from ase.units import Ang, fs
from ase.utils import reader, writer
@reader
def read_aims_output(fd, index=-1):
    """Import FHI-aims output files with all data available, i.e.
    relaxations, MD information, force information etc etc etc."""
    from ase import Atoms, Atom
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.constraints import FixAtoms, FixCartesian
    molecular_dynamics = False
    cell = []
    images = []
    fix = []
    fix_cart = []
    f = None
    pbc = False
    found_aims_calculator = False
    stress = None
    for line in fd:
        if '| Number of atoms                   :' in line:
            inp = line.split()
            n_atoms = int(inp[5])
        if '| Unit cell:' in line:
            if not pbc:
                pbc = True
                for i in range(3):
                    inp = next(fd).split()
                    cell.append([inp[1], inp[2], inp[3]])
        if 'Found relaxation constraint for atom' in line:
            xyz = [0, 0, 0]
            ind = int(line.split()[5][:-1]) - 1
            if 'All coordinates fixed' in line:
                if ind not in fix:
                    fix.append(ind)
            if 'coordinate fixed' in line:
                coord = line.split()[6]
                if coord == 'x':
                    xyz[0] = 1
                elif coord == 'y':
                    xyz[1] = 1
                elif coord == 'z':
                    xyz[2] = 1
                keep = True
                for n, c in enumerate(fix_cart):
                    if ind == c.a:
                        keep = False
                if keep:
                    fix_cart.append(FixCartesian(ind, xyz))
                else:
                    fix_cart[n].mask[xyz.index(1)] = 0
        if 'Atomic structure:' in line and (not molecular_dynamics):
            next(fd)
            atoms = Atoms()
            for _ in range(n_atoms):
                inp = next(fd).split()
                atoms.append(Atom(inp[3], (inp[4], inp[5], inp[6])))
        if 'Complete information for previous time-step:' in line:
            molecular_dynamics = True
        if 'Updated atomic structure:' in line and (not molecular_dynamics):
            atoms = _parse_atoms(fd, n_atoms=n_atoms)
        elif 'Atomic structure (and velocities)' in line:
            next(fd)
            atoms = Atoms()
            velocities = []
            for i in range(n_atoms):
                inp = next(fd).split()
                atoms.append(Atom(inp[4], (inp[1], inp[2], inp[3])))
                inp = next(fd).split()
                floatvect = [v_unit * float(l) for l in inp[1:4]]
                velocities.append(floatvect)
            atoms.set_velocities(velocities)
            if len(fix):
                atoms.set_constraint([FixAtoms(indices=fix)] + fix_cart)
            else:
                atoms.set_constraint(fix_cart)
            images.append(atoms)
        elif 'Atomic structure that was used in the preceding time step of the wrapper' in line:
            atoms = _parse_atoms(fd, n_atoms=n_atoms)
            results = images[-1].calc.results
            atoms.calc = SinglePointCalculator(atoms, **results)
            images[-1] = atoms
            atoms = atoms.copy()
        if 'Analytical stress tensor - Symmetrized' in line:
            for _ in range(4):
                next(fd)
            stress = []
            for _ in range(3):
                inp = next(fd)
                stress.append([float(i) for i in inp.split()[2:5]])
        if 'Total atomic forces' in line:
            f = []
            for i in range(n_atoms):
                inp = next(fd).split()
                f.append([float(i) for i in inp[-3:]])
            if not found_aims_calculator:
                e = images[-1].get_potential_energy()
                if stress is None:
                    calc = SinglePointCalculator(atoms, energy=e, forces=f)
                else:
                    calc = SinglePointCalculator(atoms, energy=e, forces=f, stress=stress)
                images[-1].calc = calc
            e = None
            f = None
        if 'Total energy corrected' in line:
            e = float(line.split()[5])
            if pbc:
                atoms.set_cell(cell)
                atoms.pbc = True
            if not found_aims_calculator:
                atoms.calc = SinglePointCalculator(atoms, energy=e)
            if not molecular_dynamics:
                if len(fix):
                    atoms.set_constraint([FixAtoms(indices=fix)] + fix_cart)
                else:
                    atoms.set_constraint(fix_cart)
                images.append(atoms)
            e = None
        if 'Per atom stress (eV) used for heat flux calculation' in line:
            next((l for l in fd if '-------------' in l))
            stresses = []
            for l in [next(fd) for _ in range(n_atoms)]:
                xx, yy, zz, xy, xz, yz = [float(d) for d in l.split()[2:8]]
                stresses.append([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            if not found_aims_calculator:
                e = images[-1].get_potential_energy()
                f = images[-1].get_forces()
                stress = images[-1].get_stress(voigt=False)
                calc = SinglePointCalculator(atoms, energy=e, forces=f, stress=stress, stresses=stresses)
                images[-1].calc = calc
    fd.close()
    if molecular_dynamics:
        images = images[1:]
    if isinstance(index, int):
        return images[index]
    else:
        step = index.step or 1
        if step > 0:
            start = index.start or 0
            if start < 0:
                start += len(images)
            stop = index.stop or len(images)
            if stop < 0:
                stop += len(images)
        else:
            if index.start is None:
                start = len(images) - 1
            else:
                start = index.start
                if start < 0:
                    start += len(images)
            if index.stop is None:
                stop = -1
            else:
                stop = index.stop
                if stop < 0:
                    stop += len(images)
        return [images[i] for i in range(start, stop, step)]