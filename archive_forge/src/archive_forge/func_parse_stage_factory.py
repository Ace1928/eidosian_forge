from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def parse_stage_factory(context):

    def parse(compsrc):
        source_desc = compsrc.source_desc
        full_module_name = compsrc.full_module_name
        initial_pos = (source_desc, 1, 0)
        saved_cimport_from_pyx, Options.cimport_from_pyx = (Options.cimport_from_pyx, False)
        scope = context.find_module(full_module_name, pos=initial_pos, need_pxd=0)
        Options.cimport_from_pyx = saved_cimport_from_pyx
        tree = context.parse(source_desc, scope, pxd=0, full_module_name=full_module_name)
        tree.compilation_source = compsrc
        tree.scope = scope
        tree.is_pxd = False
        return tree
    return parse