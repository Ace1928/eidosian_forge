from numba.core import (types, typing, funcdesc, config, pylowering, transforms,
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from collections import defaultdict
import warnings
def backend_object_mode():
    """
            Object mode compilation
            """
    if len(state.args) != state.nargs:
        state.args = tuple(state.args) + (types.pyobject,) * (state.nargs - len(state.args))
    return self._py_lowering_stage(state.targetctx, state.library, state.func_ir, state.flags)