from sympy.testing.pytest import warns_deprecated_sympy
def test_deprecated_utilities():
    with warns_deprecated_sympy():
        import sympy.utilities.pytest
    with warns_deprecated_sympy():
        import sympy.utilities.runtests
    with warns_deprecated_sympy():
        import sympy.utilities.randtest
    with warns_deprecated_sympy():
        import sympy.utilities.tmpfiles