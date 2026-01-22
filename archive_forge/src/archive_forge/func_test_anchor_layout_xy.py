from kivy.tests.common import GraphicUnitTest
def test_anchor_layout_xy(self):
    from kivy.uix.anchorlayout import AnchorLayout
    r = self.render
    b = self.box
    layout = AnchorLayout(anchor_y='bottom', anchor_x='left')
    layout.add_widget(b(1, 0, 0))
    r(layout)
    layout = AnchorLayout(anchor_y='top', anchor_x='right')
    layout.add_widget(b(1, 0, 0))
    r(layout)