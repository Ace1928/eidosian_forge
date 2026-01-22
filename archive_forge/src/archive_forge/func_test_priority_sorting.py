import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_priority_sorting(self):
    gdb = Recognizer()
    gdb.add_gesture('N', [NGesture], priority=10)
    gdb.add_gesture('T', [TGesture], priority=5)
    r = gdb.recognize([Ncandidate], goodscore=0.01, max_gpf=0, force_priority_sort=True)
    self.assertEqual(r.best['name'], 'T')
    r = gdb.recognize([Ncandidate], goodscore=0.01, force_priority_sort=False, max_gpf=0)
    self.assertEqual(r.best['name'], 'N')
    r = gdb.recognize([Ncandidate], goodscore=0.01, max_gpf=0, priority=10)
    self.assertEqual(r.best['name'], 'T')
    r = gdb.recognize([Ncandidate], goodscore=0.01, max_gpf=0, priority=4)
    self.assertEqual(r.best['name'], None)