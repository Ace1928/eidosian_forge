from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_feature_with_value(self):
    c = commands.FeatureCommand(b'dwim', b'please')
    self.assertEqual(b'feature dwim=please', bytes(c))