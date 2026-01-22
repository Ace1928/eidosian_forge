import pytest
from ase.build import bulk
def systems():
    yield bulk('Si')
    atoms = bulk('Fe')
    atoms.set_initial_magnetic_moments([1.0])
    yield atoms