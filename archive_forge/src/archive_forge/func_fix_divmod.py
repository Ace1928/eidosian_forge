from llvmlite import ir
def fix_divmod(mod):
    """Replace division and reminder instructions to builtins calls
    """
    _DivmodFixer().visit(mod)