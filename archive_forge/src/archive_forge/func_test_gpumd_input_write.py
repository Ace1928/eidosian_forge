import numpy as np
from ase import io
from ase.build import bulk
from ase.io.gpumd import load_xyz_input_gpumd
def test_gpumd_input_write():
    """Write a structure and read it back."""
    atoms = bulk('NiO', 'rocksalt', 4.813, cubic=True)
    atoms.write('xyz.in')
    readback = io.read('xyz.in')
    assert np.allclose(atoms.positions, readback.positions)
    assert np.allclose(atoms.cell, readback.cell)
    atoms.write('xyz.in', use_triclinic=True)
    with open('xyz.in', 'r') as fd:
        readback, input_parameters, _ = load_xyz_input_gpumd(fd)
    assert input_parameters['triclinic'] == 1
    assert np.allclose(atoms.positions, readback.positions)
    assert np.allclose(atoms.cell, readback.cell)
    assert np.array_equal(atoms.numbers, readback.numbers)
    groupings = [[[i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'Ni'], [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'O']], [[i] for i in range(len(atoms))]]
    groups = [[[j for j, group in enumerate(grouping) if i in group][0] for grouping in groupings] for i in range(len(atoms))]
    atoms.write('xyz.in', groupings=groupings)
    with open('xyz.in', 'r') as fd:
        readback, input_parameters, _ = load_xyz_input_gpumd(fd)
    assert input_parameters['num_of_groups'] == 2
    assert len(readback.info) == len(atoms)
    assert all((np.array_equal(readback.info[i]['groups'], np.array(groups[i])) for i in range(len(atoms))))
    velocities = np.array([[-0.3, 2.3, 0.7], [0.0, 0.3, 0.8], [-0.6, 0.9, 0.1], [-1.7, -0.1, -0.5], [-0.5, 0.0, 0.6], [-0.2, 0.1, 0.5], [-0.1, 1.4, -1.9], [-1.0, -0.5, -1.2]])
    atoms.set_velocities(velocities)
    atoms.write('xyz.in')
    with open('xyz.in', 'r') as fd:
        readback, input_parameters, _ = load_xyz_input_gpumd(fd)
    assert input_parameters['has_velocity'] == 1
    assert np.allclose(readback.get_velocities(), atoms.get_velocities())