import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def indexed_assign(self, list, index, value):
    list[index] = value