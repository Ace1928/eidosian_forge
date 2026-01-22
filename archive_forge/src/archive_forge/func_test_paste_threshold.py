import unittest
from collections import namedtuple
from bpython.curtsies import combined_events
from bpython.test import FixLanguageTestCase as TestCase
import curtsies.events
def test_paste_threshold(self):
    eg = EventGenerator(list('abc'))
    cb = combined_events(eg, paste_threshold=3)
    e = next(cb)
    self.assertIsInstance(e, curtsies.events.PasteEvent)
    self.assertEqual(e.events, list('abc'))
    self.assertEqual(next(cb), None)
    eg = EventGenerator(list('abc'))
    cb = combined_events(eg, paste_threshold=4)
    self.assertEqual(next(cb), 'a')
    self.assertEqual(next(cb), 'b')
    self.assertEqual(next(cb), 'c')
    self.assertEqual(next(cb), None)