import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_export_import_case_1(self):
    gdb1 = Recognizer(db=[self.Ninvar])
    gdb2 = Recognizer()
    g = gdb1.export_gesture(name='N')
    gdb2.import_gesture(g)
    r = gdb2.recognize([Ncandidate], max_gpf=0)
    self.assertEqual(r.best['name'], 'N')
    self.assertTrue(r.best['score'] > 0.94 and r.best['score'] < 0.95)