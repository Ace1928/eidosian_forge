import unittest
import os
import pytest
def test_length_simple(self):
    sound = self.get_sound()
    volume = sound.volume = 0.75
    length = sound.length
    self.assertAlmostEqual(SAMPLE_LENGTH, length, delta=DELTA)
    assert volume == sound.volume