import unittest
from traits.api import (
from traits.observation.api import (
def test_observe_extended_trait_in_list(self):
    album = Album()
    self.assertEqual(album.records_default_call_count, 0)
    self.assertEqual(len(album.record_number_change_events), 0)
    album.records[0].number += 1
    self.assertEqual(album.records_default_call_count, 1)
    self.assertEqual(len(album.record_number_change_events), 1)
    event, = album.record_number_change_events
    self.assertEqual(event.object, album.records[0])
    self.assertEqual(event.name, 'number')
    self.assertEqual(event.old, 99)
    self.assertEqual(event.new, 100)