from sympy.external import import_module
from sympy.testing.pytest import warns
def test_no_stdlib_collections3():
    """make sure we get the right collections with no catch"""
    import collections
    matplotlib = import_module('matplotlib', import_kwargs={'fromlist': ['cm', 'collections']}, min_module_version='1.1.0')
    if matplotlib:
        assert collections != matplotlib.collections