import unittest
def test_not_exists(self):
    from kivy.uix.behaviors.knspace import knspace
    self.assertRaises(AttributeError, lambda: knspace.label)