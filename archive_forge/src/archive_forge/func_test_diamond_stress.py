import pytest
import numpy as np
from ase.data.s22 import create_s22_system
from ase.build import bulk
def test_diamond_stress(factory, system):
    system = bulk('C')
    system.calc = factory.calc()
    close(system.get_potential_energy(), -0.2160072476277501)
    s_ref = np.array([0.0182329043326, 0.0182329043326, 0.0182329043326, -3.22757439831e-14, -3.2276694932e-14, -3.2276694932e-14])
    array_close(system.get_stress(), s_ref)
    s_numer = system.calc.calculate_numerical_stress(system, d=0.0001)
    array_close(s_numer, s_ref, releps=0.01, abseps=0.001)