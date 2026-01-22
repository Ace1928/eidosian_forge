import unittest
import pygame
def test_set_instrument(self):
    if not self.midi_output:
        self.skipTest('No midi device')
    out = self.midi_output
    out.set_instrument(5)
    out.set_instrument(42, channel=2)
    with self.assertRaises(ValueError) as cm:
        out.set_instrument(-6)
    self.assertEqual(str(cm.exception), 'Undefined instrument id: -6')
    with self.assertRaises(ValueError) as cm:
        out.set_instrument(156)
    self.assertEqual(str(cm.exception), 'Undefined instrument id: 156')
    with self.assertRaises(ValueError) as cm:
        out.set_instrument(5, -1)
    self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')
    with self.assertRaises(ValueError) as cm:
        out.set_instrument(5, 16)
    self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')