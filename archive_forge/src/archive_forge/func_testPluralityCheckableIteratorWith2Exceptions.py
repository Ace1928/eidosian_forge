from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import six
from six.moves import range
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
import gslib.tests.testcase as testcase
def testPluralityCheckableIteratorWith2Exceptions(self):
    """Tests PluralityCheckableIterator with 2 elements that both raise."""

    class IterTest(six.Iterator):

        def __init__(self):
            self.position = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.position < 2:
                self.position += 1
                raise CustomTestException('Test exception %s' % self.position)
            else:
                raise StopIteration()
    pcit = PluralityCheckableIterator(IterTest())
    try:
        pcit.PeekException()
        self.fail('Expected exception 1 from PeekException')
    except CustomTestException as e:
        self.assertIn(str(e), 'Test exception 1')
    try:
        for _ in pcit:
            pass
        self.fail('Expected exception 1 from iterator')
    except CustomTestException as e:
        self.assertIn(str(e), 'Test exception 1')
    try:
        pcit.PeekException()
        self.fail('Expected exception 2 from PeekException')
    except CustomTestException as e:
        self.assertIn(str(e), 'Test exception 2')
    try:
        for _ in pcit:
            pass
        self.fail('Expected exception 2 from iterator')
    except CustomTestException as e:
        self.assertIn(str(e), 'Test exception 2')
    for _ in pcit:
        self.fail('Expected StopIteration')