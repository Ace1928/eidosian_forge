import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_equal_items_not_continious(self):
    result = misc.sequence_minus([1, 2, 3, 1], [1, 3])
    self.assertEqual([2, 1], result)