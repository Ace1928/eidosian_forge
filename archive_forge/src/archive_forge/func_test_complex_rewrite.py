import unittest
def test_complex_rewrite(self):
    from kivy.lang import Builder
    from kivy.uix.widget import Widget
    from kivy.uix.label import Label
    from kivy.factory import Factory
    from kivy.properties import StringProperty, ObjectProperty, BooleanProperty
    Builder.load_string(rules)

    class TestWidget(Widget):
        source = StringProperty('')
        source2 = StringProperty('')
        source3 = StringProperty('')
        can_edit = BooleanProperty(False)

        def __init__(self, **kwargs):
            self.register_event_type('on_release')
            super(TestWidget, self).__init__(**kwargs)

        def on_release(self):
            pass

    class MainWidget(Widget):
        refwid = ObjectProperty(None)
        refwid2 = ObjectProperty(None)

    class TestWidget2(Widget):
        pass

    class CustomLabel(Label):
        pass
    Factory.register('CustomLabel', cls=CustomLabel)
    Factory.register('TestWidget', cls=TestWidget)
    Factory.register('TestWidget2', cls=TestWidget2)
    a = MainWidget()
    self.assertTrue(isinstance(a.refwid, TestWidget))
    self.assertEqual(a.refwid.source, 'valid.png')
    self.assertEqual(a.refwid.source2, 'valid.png')
    self.assertEqual(a.refwid.source3, 'valid.png')
    self.assertTrue(len(a.refwid.children) == 1)
    self.assertEqual(a.refwid.children[0].title, 'valid')
    self.assertTrue(isinstance(a.refwid2, TestWidget2))
    self.assertEqual(a.refwid2.source, 'valid.png')