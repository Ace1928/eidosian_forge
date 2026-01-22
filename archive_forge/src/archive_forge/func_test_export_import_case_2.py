import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_export_import_case_2(self):
    from tempfile import mkstemp
    import os
    gdb1 = Recognizer(db=[self.Ninvar, self.Tinvar])
    gdb2 = Recognizer()
    fh, fn = mkstemp()
    os.close(fh)
    g = gdb1.export_gesture(name='N', filename=fn)
    gdb2.import_gesture(filename=fn)
    os.unlink(fn)
    self.assertEqual(len(gdb1.db), 2)
    self.assertEqual(len(gdb2.db), 1)
    r = gdb2.recognize([Ncandidate], max_gpf=0)
    self.assertEqual(r.best['name'], 'N')
    self.assertTrue(r.best['score'] > 0.94 and r.best['score'] < 0.95)