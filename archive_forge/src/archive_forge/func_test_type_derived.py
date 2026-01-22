import unittest
from traits.api import HasTraits, Subclass, TraitError, Type
def test_type_derived(self):
    model = ExampleTypeModel(_class=DerivedClass)
    self.assertIsInstance(model._class(), DerivedClass)