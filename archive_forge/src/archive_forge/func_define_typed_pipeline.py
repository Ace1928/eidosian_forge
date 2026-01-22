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
def define_typed_pipeline(state, name='typed'):
    """Returns the typed part of the nopython pipeline"""
    pm = PassManager(name)
    pm.add_pass(NopythonTypeInference, 'nopython frontend')
    pm.add_pass(PreLowerStripPhis, 'remove phis nodes')
    pm.add_pass(InlineOverloads, 'inline overloaded functions')
    if state.flags.auto_parallel.enabled:
        pm.add_pass(PreParforPass, 'Preprocessing for parfors')
    if not state.flags.no_rewrites:
        pm.add_pass(NopythonRewrites, 'nopython rewrites')
    if state.flags.auto_parallel.enabled:
        pm.add_pass(ParforPass, 'convert to parfors')
        pm.add_pass(ParforFusionPass, 'fuse parfors')
        pm.add_pass(ParforPreLoweringPass, 'parfor prelowering')
    pm.finalize()
    return pm