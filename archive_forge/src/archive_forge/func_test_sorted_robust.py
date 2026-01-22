from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.sorting import sorted_robust, _robust_sort_keyfcn
def test_sorted_robust(self):
    a = sorted_robust([3, 2, 1])
    self.assertEqual(a, [1, 2, 3])
    a = sorted_robust([3, 2.1, 1])
    self.assertEqual(a, [1, 2.1, 3])
    a = sorted_robust([3, '2', 1])
    self.assertEqual(a, [1, 3, '2'])
    a = sorted_robust([('str1', 'str1'), (1, 'str2')])
    self.assertEqual(a, [(1, 'str2'), ('str1', 'str1')])
    a = sorted_robust([((1,), 'str2'), ('str1', 'str1')])
    self.assertEqual(a, [('str1', 'str1'), ((1,), 'str2')])
    a = sorted_robust([('str1', 'str1'), ((1,), 'str2')])
    self.assertEqual(a, [('str1', 'str1'), ((1,), 'str2')])