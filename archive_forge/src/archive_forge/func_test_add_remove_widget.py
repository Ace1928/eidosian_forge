import unittest
from tempfile import mkdtemp
from shutil import rmtree
def test_add_remove_widget(self):
    root = self.root
    self.assertEqual(root.children, [])
    c1 = self.cls()
    root.add_widget(c1)
    self.assertEqual(root.children, [c1])
    root.remove_widget(c1)
    self.assertEqual(root.children, [])