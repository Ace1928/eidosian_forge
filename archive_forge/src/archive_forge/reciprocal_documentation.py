from ase import Atoms
from ase.io import read
from ase.io.jsonio import read_json
from ase.dft.kpoints import BandPath
from ase.cli.main import CLIError
from ase.io.formats import UnknownFileTypeError
Show the reciprocal space.

    Read unit cell from a file and show a plot of the 1. Brillouin zone.  If
    the file contains information about k-points, then those can be plotted
    too.

    Examples:

        $ ase build -x fcc Al al.traj
        $ ase reciprocal al.traj
    