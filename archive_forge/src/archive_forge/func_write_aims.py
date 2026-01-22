import time
import warnings
from ase.units import Ang, fs
from ase.utils import reader, writer
@writer
def write_aims(fd, atoms, scaled=False, geo_constrain=False, velocities=False, ghosts=None, info_str=None, wrap=False):
    """Method to write FHI-aims geometry files.

    Writes the atoms positions and constraints (only FixAtoms is
    supported at the moment).

    Args:
        fd: file object
            File to output structure to
        atoms: ase.atoms.Atoms
            structure to output to the file
        scaled: bool
            If True use fractional coordinates instead of Cartesian coordinates
        symmetry_block: list of str
            List of geometric constraints as defined in:
            https://arxiv.org/abs/1908.01610
        velocities: bool
            If True add the atomic velocity vectors to the file
        ghosts: list of Atoms
            A list of ghost atoms for the system
        info_str: str
            A string to be added to the header of the file
        wrap: bool
            Wrap atom positions to cell before writing
    """
    from ase.constraints import FixAtoms, FixCartesian
    import numpy as np
    if geo_constrain:
        if not scaled:
            warnings.warn('Setting scaled to True because a symmetry_block is detected.')
            scaled = True
    fd.write('#=======================================================\n')
    if hasattr(fd, 'name'):
        fd.write('# FHI-aims file: ' + fd.name + '\n')
    fd.write('# Created using the Atomic Simulation Environment (ASE)\n')
    fd.write('# ' + time.asctime() + '\n')
    if info_str is not None:
        fd.write('\n# Additional information:\n')
        if isinstance(info_str, list):
            fd.write('\n'.join(['#  {}'.format(s) for s in info_str]))
        else:
            fd.write('# {}'.format(info_str))
        fd.write('\n')
    fd.write('#=======================================================\n')
    i = 0
    if atoms.get_pbc().any():
        for n, vector in enumerate(atoms.get_cell()):
            fd.write('lattice_vector ')
            for i in range(3):
                fd.write('%16.16f ' % vector[i])
            fd.write('\n')
    fix_cart = np.zeros([len(atoms), 3])
    if atoms.constraints:
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms):
                fix_cart[constr.index] = [1, 1, 1]
            elif isinstance(constr, FixCartesian):
                fix_cart[constr.a] = -constr.mask + 1
    if ghosts is None:
        ghosts = np.zeros(len(atoms))
    else:
        assert len(ghosts) == len(atoms)
    if geo_constrain:
        wrap = False
    scaled_positions = atoms.get_scaled_positions(wrap=wrap)
    for i, atom in enumerate(atoms):
        if ghosts[i] == 1:
            atomstring = 'empty '
        elif scaled:
            atomstring = 'atom_frac '
        else:
            atomstring = 'atom '
        fd.write(atomstring)
        if scaled:
            for pos in scaled_positions[i]:
                fd.write('%16.16f ' % pos)
        else:
            for pos in atom.position:
                fd.write('%16.16f ' % pos)
        fd.write(atom.symbol)
        fd.write('\n')
        if fix_cart[i].all():
            fd.write('    constrain_relaxation .true.\n')
        elif fix_cart[i].any():
            xyz = fix_cart[i]
            for n in range(3):
                if xyz[n]:
                    fd.write('    constrain_relaxation %s\n' % 'xyz'[n])
        if atom.charge:
            fd.write('    initial_charge %16.6f\n' % atom.charge)
        if atom.magmom:
            fd.write('    initial_moment %16.6f\n' % atom.magmom)
        if velocities and atoms.get_velocities() is not None:
            fd.write('    velocity {:.16f} {:.16f} {:.16f}\n'.format(*atoms.get_velocities()[i] / v_unit))
    if geo_constrain:
        for line in get_sym_block(atoms):
            fd.write(line)