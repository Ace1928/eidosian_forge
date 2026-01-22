import unittest
from traits.api import HasTraits, Subclass, TraitError, Type
def test_subclass_base(self):
    model = ExampleSubclassModel(_class=BaseClass)
    self.assertIsInstance(model._class(), BaseClass)