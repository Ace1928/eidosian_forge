import unittest
from traits.util.event_tracer import (
def test_exit_message_record(self):
    record = ExitMessageRecord(time=7, indent=5, handler='john', exception='sssss')
    self.assertEqual(str(record), "7 <--------- EXIT: 'john'sssss\n")
    self.assertRaises(TypeError, ExitMessageRecord, sdd=0)