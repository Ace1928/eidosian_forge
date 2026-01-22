import unittest
from kivy.tests.common import GraphicUnitTest
@unittest.skip('Currently segfault, but no idea why.')
def test_rst_replace(self):
    rst = _build_rst()
    self.render(rst)
    pg = rst.children[0].children[0].children[0]
    rendered_text = pg.text[:]
    compare_text = u'[color=202020ff][anchor=hop]ä \xa0 is [ref=None][color=ce5c00ff]replaced[/color][/ref][/color]'
    self.assertEqual(rendered_text, compare_text)