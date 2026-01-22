import unittest
import pygame
def test_write_sys_ex(self):
    if not self.midi_output:
        self.skipTest('No midi device')
    out = self.midi_output
    out.write_sys_ex(pygame.midi.time(), [240, 125, 16, 17, 18, 19, 247])