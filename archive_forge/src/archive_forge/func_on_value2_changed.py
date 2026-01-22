import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def on_value2_changed(self):
    self.obj.value2_count += 1