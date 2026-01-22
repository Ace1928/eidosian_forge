import unittest
import os
import pytest
def test_length_playing(self):
    import time
    sound = self.get_sound()
    sound.play()
    try:
        time.sleep(DELAY)
        length = sound.length
        self.assertAlmostEqual(SAMPLE_LENGTH, length, delta=DELTA)
    finally:
        sound.stop()
    self.assertAlmostEqual(SAMPLE_LENGTH, length, delta=DELTA)