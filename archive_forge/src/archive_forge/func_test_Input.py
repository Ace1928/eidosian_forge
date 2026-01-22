import unittest
import pygame
def test_Input(self):
    i = pygame.midi.get_default_input_id()
    if self.midi_input:
        self.assertEqual(self.midi_input.device_id, i)
    i = pygame.midi.get_default_output_id()
    self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, i)
    self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, 9009)
    self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, -1)
    self.assertRaises(TypeError, pygame.midi.Input, '1234')
    self.assertRaises(OverflowError, pygame.midi.Input, pow(2, 99))