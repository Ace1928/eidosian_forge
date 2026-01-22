from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import six
from six.moves import range
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
import gslib.tests.testcase as testcase
def testPluralityCheckableIteratorWith1Elem(self):
    """Tests PluralityCheckableIterator with 1 element."""
    input_list = list(range(1))
    it = iter(input_list)
    pcit = PluralityCheckableIterator(it)
    self.assertFalse(pcit.IsEmpty())
    self.assertFalse(pcit.HasPlurality())
    output_list = list(pcit)
    self.assertEqual(input_list, output_list)