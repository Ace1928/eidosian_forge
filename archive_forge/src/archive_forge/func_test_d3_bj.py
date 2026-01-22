import pytest
import numpy as np
from ase.data.s22 import create_s22_system
from ase.build import bulk
def test_d3_bj(factory, system):
    system.calc = factory.calc(damping='bj')
    close(system.get_potential_energy(), -1.211193213979179)