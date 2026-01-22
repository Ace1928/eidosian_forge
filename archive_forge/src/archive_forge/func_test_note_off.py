import unittest
import pygame
def test_note_off(self):
    if self.midi_output:
        out = self.midi_output
        out.note_on(5, 30, 0)
        out.note_off(5, 30, 0)
        with self.assertRaises(ValueError) as cm:
            out.note_off(5, 30, 25)
        self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')
        with self.assertRaises(ValueError) as cm:
            out.note_off(5, 30, -1)
        self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')