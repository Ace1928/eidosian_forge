import pytest
import numpy as np
from ase.data.s22 import create_s22_system
from ase.build import bulk
def test_d3_zero_abc(factory, system):
    system.calc = factory.calc(abc=True)
    close(system.get_potential_energy(), -0.6528640090262864)