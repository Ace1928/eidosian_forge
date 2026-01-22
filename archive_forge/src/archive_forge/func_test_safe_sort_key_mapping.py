import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_safe_sort_key_mapping(self):
    list1 = [('yellow', 1), ('blue', 2), ('red', 1)]
    data1 = self._create_dict_from_list(list1)
    list2 = [('blue', 2), ('red', 1), ('yellow', 1)]
    data2 = self._create_dict_from_list(list2)
    self.assertEqual(helpers.safe_sort_key(data1), helpers.safe_sort_key(data2))