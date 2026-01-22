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
@staticmethod
def define_nopython_pipeline(state, name='nopython'):
    """Returns an nopython mode pipeline based PassManager
        """
    dpb = DefaultPassBuilder
    pm = PassManager(name)
    untyped_passes = dpb.define_untyped_pipeline(state)
    pm.passes.extend(untyped_passes.passes)
    typed_passes = dpb.define_typed_pipeline(state)
    pm.passes.extend(typed_passes.passes)
    lowering_passes = dpb.define_nopython_lowering_pipeline(state)
    pm.passes.extend(lowering_passes.passes)
    pm.finalize()
    return pm