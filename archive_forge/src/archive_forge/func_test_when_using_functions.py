import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_when_using_functions(self):
    test_object = self.test_object
    self.assertTraitChanges(test_object, 'number', 1, test_object.add_to_number, 13.0)
    self.assertTraitDoesNotChange(test_object, 'flag', test_object.add_to_number, 13.0)