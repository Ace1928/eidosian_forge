from sympy.testing.pytest import raises
from sympy.utilities.exceptions import sympy_deprecation_warning
def test_sympy_deprecation_warning():
    raises(TypeError, lambda: sympy_deprecation_warning('test', deprecated_since_version=1.1, active_deprecations_target='active-deprecations'))
    raises(ValueError, lambda: sympy_deprecation_warning('test', deprecated_since_version='1.10', active_deprecations_target='(active-deprecations)='))