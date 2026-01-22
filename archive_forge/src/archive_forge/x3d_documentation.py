from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.utils import writer
Writes output to either an 'X3D' or an 'X3DOM' file, based on
        the extension. For X3D, filename should end in '.x3d'. For X3DOM,
        filename should end in '.html'.

        Args:
            datatype - str, output format. 'X3D' or 'X3DOM'.
        