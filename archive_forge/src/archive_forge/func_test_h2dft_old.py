import pytest
from ase.build import molecule
from ase.test.factories import ObsoleteFactoryWrapper
@pytest.mark.parametrize('name', ['aims', 'gamess_us', 'gaussian'])
def test_h2dft_old(name):
    factory = ObsoleteFactoryWrapper(name)
    run(factory)