from sympy.testing.pytest import warns_deprecated_sympy
def test_compatibility_submodule():
    with warns_deprecated_sympy():
        import sympy.core.compatibility