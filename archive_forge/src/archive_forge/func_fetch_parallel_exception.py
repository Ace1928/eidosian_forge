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
def fetch_parallel_exception(self, code):
    """
        As each OpenMP thread may raise an exception, we need to fetch that
        exception from the threadstate and save it for after the parallel
        section where it can be re-raised in the master thread.

        Although it would seem that __pyx_filename, __pyx_lineno and
        __pyx_clineno are only assigned to under exception conditions (i.e.,
        when we have the GIL), and thus should be allowed to be shared without
        any race condition, they are in fact subject to the same race
        conditions that they were previously when they were global variables
        and functions were allowed to release the GIL:

            thread A                thread B
                acquire
                set lineno
                release
                                        acquire
                                        set lineno
                                        release
                acquire
                fetch exception
                release
                                        skip the fetch

                deallocate threadstate  deallocate threadstate
        """
    code.begin_block()
    code.put_ensure_gil(declare_gilstate=True)
    code.putln_openmp('#pragma omp flush(%s)' % Naming.parallel_exc_type)
    code.putln('if (!%s) {' % Naming.parallel_exc_type)
    code.putln('__Pyx_ErrFetchWithState(&%s, &%s, &%s);' % self.parallel_exc)
    pos_info = chain(*zip(self.parallel_pos_info, self.pos_info))
    code.funcstate.uses_error_indicator = True
    code.putln('%s = %s; %s = %s; %s = %s;' % tuple(pos_info))
    code.put_gotref(Naming.parallel_exc_type, py_object_type)
    code.putln('}')
    code.put_release_ensured_gil()
    code.end_block()