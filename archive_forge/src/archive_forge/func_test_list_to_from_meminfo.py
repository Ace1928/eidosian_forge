from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_to_from_meminfo(self):
    """
        Exercise listobject.{_as_meminfo, _from_meminfo}
        """

    @njit
    def boxer():
        l = listobject.new_list(int32)
        for i in range(10, 20):
            l.append(i)
        return listobject._as_meminfo(l)
    lsttype = types.ListType(int32)

    @njit
    def unboxer(mi):
        l = listobject._from_meminfo(mi, lsttype)
        return (l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9])
    mi = boxer()
    self.assertEqual(mi.refcount, 1)
    received = list(unboxer(mi))
    expected = list(range(10, 20))
    self.assertEqual(received, expected)