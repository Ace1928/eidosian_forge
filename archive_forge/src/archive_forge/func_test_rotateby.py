import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_rotateby(self):
    r = kivy.multistroke.rotate_by(NGesture, 24)
    self.assertEqual(round(r[2].x, 1), 158.6)
    self.assertEqual(round(r[2].y, 1), 54.9)