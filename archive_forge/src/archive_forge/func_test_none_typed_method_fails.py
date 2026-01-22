import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
def test_none_typed_method_fails(self):
    """ Test that unsupported operations on List[None] raise. """

    def generate_function(line1, line2):
        context = {}
        exec(dedent('\n                from numba.typed import List\n                def bar():\n                    lst = List()\n                    {}\n                    {}\n                '.format(line1, line2)), context)
        return njit(context['bar'])
    for line1, line2 in (('lst.append(None)', 'lst.pop()'), ('lst.append(None)', 'del lst[0]'), ('lst.append(None)', 'lst.count(None)'), ('lst.append(None)', 'lst.index(None)'), ('lst.append(None)', 'lst.insert(0, None)'), ('', 'lst.insert(0, None)'), ('lst.append(None)', 'lst.clear()'), ('lst.append(None)', 'lst.copy()'), ('lst.append(None)', 'lst.extend([None])'), ('', 'lst.extend([None])'), ('lst.append(None)', 'lst.remove(None)'), ('lst.append(None)', 'lst.reverse()'), ('lst.append(None)', 'None in lst')):
        with self.assertRaises(TypingError) as raises:
            foo = generate_function(line1, line2)
            foo()
        self.assertIn('method support for List[None] is limited', str(raises.exception))