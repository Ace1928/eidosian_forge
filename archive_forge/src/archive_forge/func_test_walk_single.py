import unittest
def test_walk_single(self):
    from kivy.uix.label import Label
    label = Label()
    self.assertListEqual([n for n in label.walk(loopback=True)], [label])
    self.assertListEqual([n for n in label.walk_reverse(loopback=True)], [label])