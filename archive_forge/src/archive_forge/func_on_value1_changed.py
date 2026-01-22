import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def on_value1_changed(self):
    self.obj.value1_count += 1