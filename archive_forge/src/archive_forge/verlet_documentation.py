import numpy as np
from ase.md.md import MolecularDynamics
import warnings
Molecular Dynamics object.

        Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        timestep: float
            The time step in ASE time units.

        trajectory: Trajectory object or str  (optional)
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Default: None.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.  Default: None.

        loginterval: int (optional)
            Only write a log line for every *loginterval* time steps.  
            Default: 1

        append_trajectory: boolean
            Defaults to False, which causes the trajectory file to be
            overwriten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.

        dt: float (deprecated)
            Alias for timestep.
        