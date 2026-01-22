import unittest
from tempfile import mkdtemp
from shutil import rmtree
def test_clear_widgets(self):
    root = self.root
    self.assertEqual(root.children, [])
    c1 = self.cls()
    c2 = self.cls()
    c3 = self.cls()
    root.add_widget(c1, index=0)
    root.add_widget(c2, index=1)
    root.add_widget(c3, index=2)
    self.assertEqual(root.children, [c1, c2, c3])
    root.clear_widgets([c2])
    self.assertEqual(root.children, [c1, c3])
    root.clear_widgets([])
    self.assertEqual(root.children, [c1, c3])
    root.clear_widgets()
    self.assertEqual(root.children, [])