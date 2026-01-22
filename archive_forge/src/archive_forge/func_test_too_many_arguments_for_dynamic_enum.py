import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
def test_too_many_arguments_for_dynamic_enum(self):
    with self.assertRaises(TraitError):
        Enum('red', 'green', values='values')