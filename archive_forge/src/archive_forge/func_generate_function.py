from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def generate_function(line):
    context = {}
    exec(dedent('\n                from numba.typed import listobject\n                from numba import int32\n                def bar():\n                    lst = listobject.new_list(int32)\n                    lst.append(int32(1))\n                    lst._make_immutable()\n                    zero = int32(0)\n                    {}\n                '.format(line)), context)
    return njit(context['bar'])