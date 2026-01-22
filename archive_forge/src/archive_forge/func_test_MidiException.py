import unittest
import pygame
def test_MidiException(self):
    """Ensures the MidiException is raised as expected."""

    def raiseit():
        raise pygame.midi.MidiException('Hello Midi param')
    with self.assertRaises(pygame.midi.MidiException) as cm:
        raiseit()
    self.assertEqual(cm.exception.parameter, 'Hello Midi param')