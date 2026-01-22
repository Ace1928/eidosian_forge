from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def setup_parallel_control_flow_block(self, code):
    """
        Sets up a block that surrounds the parallel block to determine
        how the parallel section was exited. Any kind of return is
        trapped (break, continue, return, exceptions). This is the idea:

        {
            int why = 0;

            #pragma omp parallel
            {
                return # -> goto new_return_label;
                goto end_parallel;

            new_return_label:
                why = 3;
                goto end_parallel;

            end_parallel:;
                #pragma omp flush(why) # we need to flush for every iteration
            }

            if (why == 3)
                goto old_return_label;
        }
        """
    self.old_loop_labels = code.new_loop_labels()
    self.old_error_label = code.new_error_label()
    self.old_return_label = code.return_label
    code.return_label = code.new_label(name='return')
    code.begin_block()
    self.begin_of_parallel_control_block_point = code.insertion_point()
    self.begin_of_parallel_control_block_point_after_decls = code.insertion_point()
    self.undef_builtin_expect_apple_gcc_bug(code)