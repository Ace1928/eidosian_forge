import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_some_items_are_equal(self):
    result = misc.sequence_minus([1, 1, 1, 1], [1, 1, 3])
    self.assertEqual([1, 1], result)