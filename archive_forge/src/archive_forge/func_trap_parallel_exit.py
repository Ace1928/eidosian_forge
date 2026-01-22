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
def trap_parallel_exit(self, code, should_flush=False):
    """
        Trap any kind of return inside a parallel construct. 'should_flush'
        indicates whether the variable should be flushed, which is needed by
        prange to skip the loop. It also indicates whether we need to register
        a continue (we need this for parallel blocks, but not for prange
        loops, as it is a direct jump there).

        It uses the same mechanism as try/finally:
            1 continue
            2 break
            3 return
            4 error
        """
    save_lastprivates_label = code.new_label()
    dont_return_label = code.new_label()
    self.any_label_used = False
    self.breaking_label_used = False
    self.error_label_used = False
    self.parallel_private_temps = []
    all_labels = code.get_all_labels()
    for label in all_labels:
        if code.label_used(label):
            self.breaking_label_used = self.breaking_label_used or label != code.continue_label
            self.any_label_used = True
    if self.any_label_used:
        code.put_goto(dont_return_label)
    for i, label in enumerate(all_labels):
        if not code.label_used(label):
            continue
        is_continue_label = label == code.continue_label
        code.put_label(label)
        if not (should_flush and is_continue_label):
            if label == code.error_label:
                self.error_label_used = True
                self.fetch_parallel_exception(code)
            code.putln('%s = %d;' % (Naming.parallel_why, i + 1))
        if self.breaking_label_used and self.is_prange and (not is_continue_label):
            code.put_goto(save_lastprivates_label)
        else:
            code.put_goto(dont_return_label)
    if self.any_label_used:
        if self.is_prange and self.breaking_label_used:
            code.put_label(save_lastprivates_label)
            self.save_parallel_vars(code)
        code.put_label(dont_return_label)
        if should_flush and self.breaking_label_used:
            code.putln_openmp('#pragma omp flush(%s)' % Naming.parallel_why)