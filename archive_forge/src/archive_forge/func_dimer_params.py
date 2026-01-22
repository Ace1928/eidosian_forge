import pytest
@pytest.fixture
def dimer_params():
    dimer_params = {}
    a = 2.0
    dimer_params['symbols'] = 'Ni' * 2
    dimer_params['positions'] = [(0, 0, 0), (a, 0, 0)]
    dimer_params['cell'] = (1000 * a, 1000 * a, 1000 * a)
    dimer_params['pbc'] = (False, False, False)
    return dimer_params