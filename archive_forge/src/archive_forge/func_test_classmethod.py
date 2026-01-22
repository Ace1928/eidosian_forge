import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_classmethod(self):

    class WithClassMethod:

        @classmethod
        def classfunc(cls):
            pass
    self.assertTrue(_callable(WithClassMethod.classfunc))