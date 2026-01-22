import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
def test_enum_enum(self):
    example = EnumEnumExample()
    self.assertEqual(example.value, FooEnum.foo)
    self.assertEqual(example.value_default, FooEnum.bar)
    self.assertEqual(example.value_name, FooEnum.foo)
    self.assertEqual(example.value_name_default, FooEnum.bar)
    example.value = FooEnum.bar
    self.assertEqual(example.value, FooEnum.bar)
    with self.assertRaises(TraitError):
        example.value = 'foo'
    with self.assertRaises(TraitError):
        example.value = 0
    example.values = OtherEnum
    example.value_name = OtherEnum.two
    self.assertEqual(example.value_name, OtherEnum.two)
    with self.assertRaises(TraitError):
        example.value_name = FooEnum.bar