import pytest
from ase.io.cif import CIFBlock, parse_loop, CIFLoop
@pytest.fixture
def cifblock():
    return CIFBlock('hello', {'_cifkey': 42})