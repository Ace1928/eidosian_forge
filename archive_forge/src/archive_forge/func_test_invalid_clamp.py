import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_invalid_clamp(self):
    self.assertRaises(ValueError, misc.clamp, 0.0, 2.0, 1.0)