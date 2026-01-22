from sympy.external import import_module
def src_to_sympy(src):
    """Wrapper function to convert the given Fortran source code to SymPy Expressions

    Parameters
    ==========

    src : string
        A string with the Fortran source code

    Returns
    =======

    py_src : string
        A string with the Python source code compatible with SymPy

    """
    a_ast = src_to_ast(src, translation_unit=False)
    a = ast_to_asr(a_ast)
    py_src = call_visitor(a)
    return py_src