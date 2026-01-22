from ase.atoms import Atoms
from ase.utils import reader, writer
@writer
def write_crystal(fd, atoms):
    """Method to write atom structure in crystal format
       (fort.34 format)
    """
    ispbc = atoms.get_pbc()
    box = atoms.get_cell()
    if ispbc[2]:
        fd.write('%2s %2s %2s %23s \n' % ('3', '1', '1', 'E -0.0E+0 DE 0.0E+0( 1)'))
    elif ispbc[1]:
        fd.write('%2s %2s %2s %23s \n' % ('2', '1', '1', 'E -0.0E+0 DE 0.0E+0( 1)'))
        box[2, 2] = 500.0
    elif ispbc[0]:
        fd.write('%2s %2s %2s %23s \n' % ('1', '1', '1', 'E -0.0E+0 DE 0.0E+0( 1)'))
        box[2, 2] = 500.0
        box[1, 1] = 500.0
    else:
        fd.write('%2s %2s %2s %23s \n' % ('0', '1', '1', 'E -0.0E+0 DE 0.0E+0( 1)'))
        box[2, 2] = 500.0
        box[1, 1] = 500.0
        box[0, 0] = 500.0
    fd.write(' %.17E %.17E %.17E \n' % (box[0][0], box[0][1], box[0][2]))
    fd.write(' %.17E %.17E %.17E \n' % (box[1][0], box[1][1], box[1][2]))
    fd.write(' %.17E %.17E %.17E \n' % (box[2][0], box[2][1], box[2][2]))
    fd.write(' %2s \n' % 1)
    fd.write(' %.17E %.17E %.17E \n' % (1, 0, 0))
    fd.write(' %.17E %.17E %.17E \n' % (0, 1, 0))
    fd.write(' %.17E %.17E %.17E \n' % (0, 0, 1))
    fd.write(' %.17E %.17E %.17E \n' % (0, 0, 0))
    fd.write(' %8s \n' % len(atoms))
    coords = atoms.get_positions()
    tags = atoms.get_tags()
    atomnum = atoms.get_atomic_numbers()
    for iatom, coord in enumerate(coords):
        fd.write('%5i  %19.16f %19.16f %19.16f \n' % (atomnum[iatom] + tags[iatom], coords[iatom][0], coords[iatom][1], coords[iatom][2]))