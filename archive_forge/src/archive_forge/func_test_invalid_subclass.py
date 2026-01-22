import unittest
from traits.api import HasTraits, Subclass, TraitError, Type
def test_invalid_subclass(self):
    example_model = ExampleSubclassModel()

    def assign_invalid():
        example_model._class = UnrelatedClass
    self.assertRaises(TraitError, assign_invalid)