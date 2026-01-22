import unittest
from traits.api import (
from traits.observation.api import (
def test_observe_extended_trait_in_default_dict(self):
    album = Album()
    self.assertEqual(album.name_to_records_default_call_count, 0)
    self.assertEqual(len(album.name_to_records_clicked_events), 0)
    album.name_to_records['Record'].clicked = True
    self.assertEqual(len(album.name_to_records_clicked_events), 1)