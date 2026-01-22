import unittest
import os
import pytest
def test_length_stopped(self):
    import time
    sound = self.get_sound()
    sound.play()
    try:
        time.sleep(DELAY)
    finally:
        sound.stop()
    length = sound.length
    self.assertAlmostEqual(SAMPLE_LENGTH, length, delta=DELTA)