from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def create_pyx_pipeline(context, options, result, py=False, exclude_classes=()):
    mode = 'py' if py else 'pyx'
    test_support = []
    ctest_support = []
    if options.evaluate_tree_assertions:
        from ..TestUtils import TreeAssertVisitor
        test_validator = TreeAssertVisitor()
        test_support.append(test_validator)
        ctest_support.append(test_validator.create_c_file_validator())
    if options.gdb_debug:
        from ..Debugger import DebugWriter
        from .ParseTreeTransforms import DebugTransform
        context.gdb_debug_outputwriter = DebugWriter.CythonDebugWriter(options.output_dir)
        debug_transform = [DebugTransform(context, options, result)]
    else:
        debug_transform = []
    return list(itertools.chain([parse_stage_factory(context)], create_pipeline(context, mode, exclude_classes=exclude_classes), test_support, [inject_pxd_code_stage_factory(context), inject_utility_code_stage_factory(context), abort_on_errors], debug_transform, [generate_pyx_code_stage_factory(options, result)], ctest_support))