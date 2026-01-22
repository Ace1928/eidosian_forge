import io
import re
import pathlib
import numpy as np
from ase.calculators.lammps import convert
def extract_cell(raw_datafile_contents):
    """
    NOTE: Assumes an orthogonal cell (xy, xz, yz tilt factors are
    ignored even if they exist)
    """
    RE_CELL = re.compile('\n            (\\S+)\\s+(\\S+)\\s+xlo\\s+xhi\\n\n            (\\S+)\\s+(\\S+)\\s+ylo\\s+yhi\\n\n            (\\S+)\\s+(\\S+)\\s+zlo\\s+zhi\\n\n        ', flags=re.VERBOSE)
    xlo, xhi, ylo, yhi, zlo, zhi = map(float, RE_CELL.search(raw_datafile_contents).groups())
    cell = [[xhi - xlo, 0, 0], [0, yhi - ylo, 0], [0, 0, zhi - zlo]]
    return cell