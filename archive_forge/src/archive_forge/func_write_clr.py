import numpy as np
import ase
from ase.data import chemical_symbols
from ase.utils import reader, writer
def write_clr(fd, atoms):
    """Write extra color and radius code to a CLR-file (for use with AtomEye).
       Hit F12 in AtomEye to use.
       See: http://mt.seas.upenn.edu/Archive/Graphics/A/
    """
    color = None
    radius = None
    if atoms.has('color'):
        color = atoms.get_array('color')
    if atoms.has('radius'):
        radius = atoms.get_array('radius')
    if color is None:
        color = np.zeros([len(atoms), 3], dtype=float)
        for a in atoms:
            color[a.index, :] = default_color[a.symbol]
    if radius is None:
        radius = np.zeros(len(atoms), dtype=float)
        for a in atoms:
            radius[a.index] = default_radius[a.symbol]
    radius.shape = (-1, 1)
    for c1, c2, c3, r in np.append(color, radius, axis=1):
        fd.write('%f %f %f %f\n' % (c1, c2, c3, r))