import pytest
from ase.utils import deprecated, devnull, tokenize_version
def test_deprecated_devnull():
    with pytest.warns(DeprecationWarning):
        devnull.tell()