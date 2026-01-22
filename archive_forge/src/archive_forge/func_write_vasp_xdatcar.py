import re
import numpy as np
from ase import Atoms
from ase.utils import reader, writer
from ase.io.utils import ImageIterator
from ase.io import ParseError
from .vasp_parsers import vasp_outcar_parsers as vop
from pathlib import Path
@writer
def write_vasp_xdatcar(fd, images, label=None):
    """Write VASP MD trajectory (XDATCAR) file

    Only Vasp 5 format is supported (for consistency with read_vasp_xdatcar)

    Args:
        fd (str, fp): Output file
        images (iterable of Atoms): Atoms images to write. These must have
            consistent atom order and lattice vectors - this will not be
            checked.
        label (str): Text for first line of file. If empty, default to list of
            elements.

    """
    images = iter(images)
    image = next(images)
    if not isinstance(image, Atoms):
        raise TypeError('images should be a sequence of Atoms objects.')
    symbol_count = _symbol_count_from_symbols(image.get_chemical_symbols())
    if label is None:
        label = ' '.join([s for s, _ in symbol_count])
    fd.write(label + '\n')
    fd.write('           1\n')
    float_string = '{:11.6f}'
    for row_i in range(3):
        fd.write('  ')
        fd.write(' '.join((float_string.format(x) for x in image.cell[row_i])))
        fd.write('\n')
    _write_symbol_count(fd, symbol_count)
    _write_xdatcar_config(fd, image, index=1)
    for i, image in enumerate(images):
        _write_xdatcar_config(fd, image, i + 2)