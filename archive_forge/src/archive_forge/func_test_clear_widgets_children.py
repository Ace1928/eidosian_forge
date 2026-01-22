import unittest
from tempfile import mkdtemp
from shutil import rmtree
def test_clear_widgets_children(self):
    root = self.root
    for _ in range(10):
        root.add_widget(self.cls())
    self.assertEqual(len(root.children), 10)
    root.clear_widgets(root.children)
    self.assertEqual(root.children, [])