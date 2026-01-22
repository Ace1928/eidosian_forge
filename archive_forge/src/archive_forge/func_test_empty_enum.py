import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
def test_empty_enum(self):
    with self.assertRaises(TraitError):

        class EmptyEnum(HasTraits):
            a = Enum()
        EmptyEnum()