from llvmlite.ir import CallInstr
def replace_all_calls(mod, orig, repl):
    """Replace all calls to `orig` to `repl` in module `mod`.
    Returns the references to the returned calls
    """
    rc = ReplaceCalls(orig, repl)
    rc.visit(mod)
    return rc.calls