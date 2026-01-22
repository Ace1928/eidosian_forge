import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_expected_count_custom_decr(self):
    upper = 100
    it = misc.countdown_iter(upper, decr=2)
    items = []
    for i in it:
        self.assertEqual(upper, i)
        upper -= 2
        items.append(i)
    self.assertEqual(0, upper)
    self.assertEqual(50, len(items))