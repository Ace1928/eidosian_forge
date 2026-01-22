import unittest
from tempfile import mkdtemp
from shutil import rmtree
def test_position(self):
    wid = self.root
    wid.x = 50
    self.assertEqual(wid.x, 50)
    self.assertEqual(wid.pos, [50, 0])
    wid.y = 60
    self.assertEqual(wid.y, 60)
    self.assertEqual(wid.pos, [50, 60])
    wid.pos = (0, 0)
    self.assertEqual(wid.pos, [0, 0])
    self.assertEqual(wid.x, 0)
    self.assertEqual(wid.y, 0)