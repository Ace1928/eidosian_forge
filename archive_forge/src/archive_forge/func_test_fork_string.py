import unittest
def test_fork_string(self):
    from kivy.lang import Builder
    from kivy.uix.behaviors.knspace import knspace
    w = Builder.load_string("\n<NamedLabel@KNSpaceBehavior+Label>\n\nBoxLayout:\n    NamedLabel:\n        knspace: 'fork'\n        knsname: 'label9'\n        text: 'Hello'\n    NamedLabel:\n        knspace: 'fork'\n        knsname: 'label9'\n        text: 'Goodbye'\n")
    self.assertEqual(w.children[0].knspace.label9.text, 'Goodbye')
    self.assertEqual(w.children[1].knspace.label9.text, 'Hello')