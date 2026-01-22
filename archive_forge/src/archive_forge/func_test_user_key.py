from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.sorting import sorted_robust, _robust_sort_keyfcn
def test_user_key(self):
    sorted_robust([(('10_1', 2), None), ((10, 2), None)], key=lambda x: x[0])