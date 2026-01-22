from collections import namedtuple
import copy
import warnings
from numba.core.tracing import event
from numba.core import (utils, errors, interpreter, bytecode, postproc, config,
from numba.parfors.parfor import ParforDiagnostics
from numba.core.errors import CompilerError
from numba.core.environment import lookup_environment
from numba.core.compiler_machinery import PassManager
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.object_mode_passes import (ObjectModeFrontEnd,
from numba.core.targetconfig import TargetConfig, Option, ConfigStack
def compile_ir(self, func_ir, lifted=(), lifted_from=None):
    self.state.func_id = func_ir.func_id
    self.state.lifted = lifted
    self.state.lifted_from = lifted_from
    self.state.func_ir = func_ir
    self.state.nargs = self.state.func_ir.arg_count
    FixupArgs().run_pass(self.state)
    return self._compile_ir()