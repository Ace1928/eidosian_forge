import unittest
from unittest import mock
from traits.api import HasTraits, Int
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._testing import (
def test_metadata_defined_vs_undefined(self):
    metadata_filter = MetadataFilter(metadata_name='name')
    self.assertTrue(metadata_filter('name', Int(name=True).as_ctrait()), 'Expected the filter to return true')
    self.assertFalse(metadata_filter('name', Int().as_ctrait()), 'Expected the filter to return false')