import unittest
import warnings
from traits.api import (
def bar_changed(self, object, trait, old, new):
    self.changed_object = object
    self.changed_trait = trait
    self.changed_old = old
    self.changed_new = new
    self.changed_count += 1