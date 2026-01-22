import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
def test_explicit_collection_with_no_elements(self):
    with self.assertRaises(TraitError):
        Enum([])
    with self.assertRaises(TraitError):
        Enum(3.5, [])