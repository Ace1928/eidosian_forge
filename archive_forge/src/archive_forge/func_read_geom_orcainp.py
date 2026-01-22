from io import StringIO
from ase.io import read
from ase.utils import reader
@reader
def read_geom_orcainp(fd):
    """Method to read geometry from an ORCA input file."""
    lines = fd.readlines()
    stopline = 0
    for index, line in enumerate(lines):
        if line[1:].startswith('xyz '):
            startline = index + 1
            stopline = -1
        elif line.startswith('end') and stopline == -1:
            stopline = index
        elif line.startswith('*') and stopline == -1:
            stopline = index
    xyz_text = '%i\n' % (stopline - startline)
    xyz_text += ' geometry\n'
    for line in lines[startline:stopline]:
        xyz_text += line
    atoms = read(StringIO(xyz_text), format='xyz')
    atoms.set_cell((0.0, 0.0, 0.0))
    return atoms