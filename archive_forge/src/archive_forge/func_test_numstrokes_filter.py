import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_numstrokes_filter(self):
    gdb = Recognizer(db=[self.Ninvar, self.Nbound])
    n = gdb.filter(numstrokes=2)
    self.assertEqual(len(n), 0)
    gdb.add_gesture('T', [TGesture, TGesture])
    n = gdb.filter(numstrokes=2)
    self.assertEqual(len(n), 1)
    n = gdb.filter(numstrokes=[1, 2])
    self.assertEqual(len(n), 3)