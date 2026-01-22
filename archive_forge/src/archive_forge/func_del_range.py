import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def del_range(self, list, index1, index2):
    del list[index1:index2]