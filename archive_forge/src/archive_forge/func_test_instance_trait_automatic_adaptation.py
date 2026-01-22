import unittest
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
def test_instance_trait_automatic_adaptation(self):
    bar = Bar()
    foo_container = self.create_foo_container()
    with self.assertRaises(TraitError):
        foo_container.not_adapting_foo = bar
    with self.assertRaises(TraitError):
        foo_container.adapting_foo = bar
    foo_container.adapting_foo_permissive = bar
    self.assertIsNone(foo_container.adapting_foo_permissive)
    foo_container.adapting_foo_dynamic_default = bar
    self.assertIsInstance(foo_container.adapting_foo_dynamic_default, Foo)
    self.assertTrue(foo_container.adapting_foo_dynamic_default.default)
    register_factory(bar_to_foo_adapter, Bar, Foo)
    with self.assertRaises(TraitError):
        foo_container.not_adapting_foo = bar
    foo_container.adapting_foo = bar
    self.assertIsInstance(foo_container.adapting_foo, Foo)
    foo_container.adapting_foo_permissive = bar
    self.assertIsInstance(foo_container.adapting_foo_permissive, Foo)
    foo_container.adapting_foo_dynamic_default = bar
    self.assertIsInstance(foo_container.adapting_foo_dynamic_default, Foo)
    self.assertFalse(foo_container.adapting_foo_dynamic_default.default)