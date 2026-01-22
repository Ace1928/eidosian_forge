import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_documented_property(self):

    class A(object):

        @misc.cachedproperty
        def b(self):
            """I like bees."""
            return 'b'
    self.assertEqual('I like bees.', inspect.getdoc(A.b))