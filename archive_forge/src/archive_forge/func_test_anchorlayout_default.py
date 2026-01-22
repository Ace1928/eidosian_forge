from kivy.tests.common import GraphicUnitTest
def test_anchorlayout_default(self):
    from kivy.uix.anchorlayout import AnchorLayout
    r = self.render
    b = self.box
    layout = AnchorLayout()
    layout.add_widget(b(1, 0, 0))
    r(layout)