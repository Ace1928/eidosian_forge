import unittest
import pygame
def test_get_default_input_id(self):
    midin_id = pygame.midi.get_default_input_id()
    self.assertIsInstance(midin_id, int)
    self.assertTrue(midin_id >= -1)
    pygame.midi.quit()
    self.assertRaises(RuntimeError, pygame.midi.get_default_output_id)