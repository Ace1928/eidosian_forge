import unittest
def test_allow_none(self):
    from kivy.uix.behaviors.knspace import knspace, KNSpaceBehavior
    from kivy.uix.widget import Widget

    class MyWidget(KNSpaceBehavior, Widget):
        pass
    knspace.label3 = 1
    knspace.label3 = None
    w = MyWidget()
    w.knspace = knspace
    w.knspace = None