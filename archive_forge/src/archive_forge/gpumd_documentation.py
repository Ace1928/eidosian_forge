import numpy as np
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase import Atoms

    Read Atoms object from a GPUMD structure input file

    Parameters
    ----------
    fd : file | str
        File object or name of file from which to read the Atoms object
    species : List[str]
        List with the chemical symbols that correspond to each type, will take
        precedence over isotope_masses
    isotope_masses: Dict[str, List[float]]
        Dictionary with chemical symbols and lists of the associated atomic
        masses, which is used to identify the chemical symbols that correspond
        to the types not found in species_types. The default is to find the
        closest match :data:`ase.data.atomic_masses`.

    Returns
    -------
    atoms : Atoms
        Atoms object

    Raises
    ------
    ValueError
        Raised if the list of species is incompatible with the input file
    