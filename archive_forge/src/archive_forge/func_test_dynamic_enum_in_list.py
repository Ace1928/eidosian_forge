import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
def test_dynamic_enum_in_list(self):

    class HasEnumInList(HasTraits):
        digits = Set(Int)
        digit_sequence = List(Enum(values='digits'))
    model = HasEnumInList(digits={-1, 0, 1})
    model.digit_sequence = [-1, 0, 1, 1]
    with self.assertRaises(TraitError):
        model.digit_sequence = [-1, 0, 2, 1]