import pytest
from ase.utils import deprecated, devnull, tokenize_version
@pytest.mark.parametrize('v1, v2', [('1', '2'), ('a', 'b'), ('9.0', '10.0'), ('3.8.0', '3.8.1'), ('3a', '3b'), ('3', '3a')])
def test_tokenize_version_lessthan(v1, v2):
    v1 = tokenize_version(v1)
    v2 = tokenize_version(v2)
    assert v1 < v2