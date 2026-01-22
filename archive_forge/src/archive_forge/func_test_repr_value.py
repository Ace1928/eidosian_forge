import unittest
from unittest import mock
from traits.api import HasTraits, Int
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._testing import (
def test_repr_value(self):
    metadata_filter = MetadataFilter(metadata_name='name')
    actual = repr(metadata_filter)
    self.assertEqual(actual, "MetadataFilter(metadata_name='name')")