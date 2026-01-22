import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@property
def atom_style(self):
    """
        See if a 'model-init' field exists in the SM metadata and, if
        so, whether it contains any entries including an "atom_style"
        command.  This is specific to LAMMPS SMs and is only required
        for using the LAMMPSrun calculator because it uses
        lammps.inputwriter to create a data file.  All other content in
        'model-init', if it exists, is ignored.
        """
    atom_style = None
    for ln in self.metadata.get('model-init', []):
        if ln.find('atom_style') != -1:
            atom_style = ln.split()[1]
    return atom_style