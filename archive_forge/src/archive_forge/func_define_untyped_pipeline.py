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
def define_untyped_pipeline(state, name='untyped'):
    """Returns an untyped part of the nopython pipeline"""
    pm = PassManager(name)
    if config.USE_RVSDG_FRONTEND:
        if state.func_ir is None:
            pm.add_pass(RVSDGFrontend, 'rvsdg frontend')
            pm.add_pass(FixupArgs, 'fix up args')
        pm.add_pass(IRProcessing, 'processing IR')
    else:
        if state.func_ir is None:
            pm.add_pass(TranslateByteCode, 'analyzing bytecode')
            pm.add_pass(FixupArgs, 'fix up args')
        pm.add_pass(IRProcessing, 'processing IR')
    pm.add_pass(WithLifting, 'Handle with contexts')
    pm.add_pass(InlineClosureLikes, 'inline calls to locally defined closures')
    if not state.flags.no_rewrites:
        pm.add_pass(RewriteSemanticConstants, 'rewrite semantic constants')
        pm.add_pass(DeadBranchPrune, 'dead branch pruning')
        pm.add_pass(GenericRewrites, 'nopython rewrites')
    pm.add_pass(RewriteDynamicRaises, 'rewrite dynamic raises')
    pm.add_pass(MakeFunctionToJitFunction, 'convert make_function into JIT functions')
    pm.add_pass(InlineInlinables, 'inline inlinable functions')
    if not state.flags.no_rewrites:
        pm.add_pass(DeadBranchPrune, 'dead branch pruning')
    pm.add_pass(FindLiterallyCalls, 'find literally calls')
    pm.add_pass(LiteralUnroll, 'handles literal_unroll')
    if state.flags.enable_ssa:
        pm.add_pass(ReconstructSSA, 'ssa')
    pm.add_pass(LiteralPropagationSubPipelinePass, 'Literal propagation')
    pm.finalize()
    return pm