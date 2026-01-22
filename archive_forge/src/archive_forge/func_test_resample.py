import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_resample(self):
    r = kivy.multistroke.resample([Vector(0, 0), Vector(1, 1)], 11)
    self.assertEqual(len(r), 11)
    self.assertEqual(round(r[9].x, 1), 0.9)
    r = kivy.multistroke.resample(TGesture, 25)
    self.assertEqual(len(r), 25)
    self.assertEqual(round(r[12].x), 81)
    self.assertEqual(r[12].y, 7)
    self.assertEqual(TGesture[3].x, r[24].x)
    self.assertEqual(TGesture[3].y, r[24].y)