import pytest
from ase.utils import deprecated, devnull, tokenize_version
def test_deprecated_decorator():

    class MyWarning(UserWarning):
        pass

    @deprecated('hello', MyWarning)
    def add(a, b):
        return a + b
    with pytest.warns(MyWarning, match='hello'):
        assert add(2, 2) == 4