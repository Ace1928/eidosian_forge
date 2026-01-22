import numpy as np
from ase import io
from ase.build import bulk
from ase.io.gpumd import load_xyz_input_gpumd
def test_load_gpumd_input():
    """Load all information from a GPUMD input file."""
    with open('xyz.in', 'w') as fd:
        fd.write(gpumd_input_text)
    species_ref = ['Si', 'C']
    with open('xyz.in', 'r') as fd:
        atoms, input_parameters, species = load_xyz_input_gpumd(fd, species=species_ref)
    input_parameters_ref = {'N': 16, 'M': 4, 'cutoff': 1.1, 'triclinic': 0, 'has_velocity': 1, 'num_of_groups': 2}
    assert input_parameters == input_parameters_ref
    assert species == species_ref