import numpy as np
from ase.atoms import Atoms
from ase.units import Hartree
from ase.data import atomic_numbers
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import writer, reader
@reader
def iread_xsf(fileobj, read_data=False):
    """Yield images and optionally data from xsf file.

    Yields image1, image2, ..., imageN[, data].

    Images are Atoms objects and data is a numpy array.

    Presently supports only a single 3D datagrid."""

    def _line_generator_func():
        for line in fileobj:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            yield line
    _line_generator = _line_generator_func()

    def readline():
        return next(_line_generator)
    line = readline()
    if line.startswith('ANIMSTEPS'):
        nimages = int(line.split()[1])
        line = readline()
    else:
        nimages = 1
    if line == 'CRYSTAL':
        pbc = (True, True, True)
    elif line == 'SLAB':
        pbc = (True, True, False)
    elif line == 'POLYMER':
        pbc = (True, False, False)
    else:
        assert line.startswith('ATOMS'), line
        pbc = (False, False, False)
    cell = None
    for n in range(nimages):
        if any(pbc):
            line = readline()
            if line.startswith('PRIMCOORD'):
                assert cell is not None
            else:
                assert line.startswith('PRIMVEC')
                cell = []
                for i in range(3):
                    cell.append([float(x) for x in readline().split()])
                line = readline()
                if line.startswith('CONVVEC'):
                    for i in range(3):
                        readline()
                    line = readline()
            assert line.startswith('PRIMCOORD')
            natoms = int(readline().split()[0])
            lines = [readline() for _ in range(natoms)]
        else:
            assert line.startswith('ATOMS'), line
            line = readline()
            lines = []
            while not (line.startswith('ATOMS') or line.startswith('BEGIN')):
                lines.append(line)
                try:
                    line = readline()
                except StopIteration:
                    break
            if line.startswith('BEGIN'):
                data_header_line = line
        numbers = []
        positions = []
        for positionline in lines:
            tokens = positionline.split()
            symbol = tokens[0]
            if symbol.isdigit():
                numbers.append(int(symbol))
            else:
                numbers.append(atomic_numbers[symbol.capitalize()])
            positions.append([float(x) for x in tokens[1:]])
        positions = np.array(positions)
        if len(positions[0]) == 3:
            forces = None
        else:
            forces = positions[:, 3:] * Hartree
            positions = positions[:, :3]
        image = Atoms(numbers, positions, cell=cell, pbc=pbc)
        if forces is not None:
            image.calc = SinglePointCalculator(image, forces=forces)
        yield image
    if read_data:
        if any(pbc):
            line = readline()
        else:
            line = data_header_line
        assert line.startswith('BEGIN_BLOCK_DATAGRID_3D')
        readline()
        line = readline()
        assert line.startswith('BEGIN_DATAGRID_3D')
        shape = [int(x) for x in readline().split()]
        assert len(shape) == 3
        readline()
        for i in range(3):
            readline()
        npoints = np.prod(shape)
        data = []
        line = readline()
        while not line.startswith('END_DATAGRID_3D'):
            data.extend([float(x) for x in line.split()])
            line = readline()
        assert len(data) == npoints
        data = np.array(data, float).reshape(shape[::-1]).T
        yield data