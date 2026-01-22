import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_call_magic_method(self):

    class Callable:

        def __call__(self):
            pass
    instance = Callable()
    self.assertTrue(_callable(instance))