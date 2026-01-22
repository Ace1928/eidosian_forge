import os
import sys
import tempfile
import textwrap
import shutil
import subprocess
import unittest
from traits.api import (
from traits.testing.optional_dependencies import requires_numpy
def test_disallowed_default_value(self):

    class MyTraitType(TraitType):
        default_value_type = DefaultValue.disallow
    trait_type = MyTraitType()
    self.assertEqual(trait_type.get_default_value(), (DefaultValue.disallow, Undefined))
    ctrait = trait_type.as_ctrait()
    self.assertEqual(ctrait.default_value(), (DefaultValue.disallow, Undefined))
    self.assertEqual(ctrait.default_kind, 'invalid')
    self.assertEqual(ctrait.default, Undefined)
    with self.assertRaises(ValueError):
        ctrait.default_value_for(None, '<dummy>')