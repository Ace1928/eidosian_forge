from __future__ import unicode_literals
import unittest
import commonmark
from commonmark.blocks import Parser
from commonmark.render.html import HtmlRenderer
from commonmark.inlines import InlineParser
from commonmark.node import NodeWalker, Node
def test_smart_dashes(self):
    md = 'a - b -- c --- d ---- e ----- f'
    EM = '—'
    EN = '–'
    expected_html = '<p>' + 'a - ' + 'b ' + EN + ' ' + 'c ' + EM + ' ' + 'd ' + EN + EN + ' ' + 'e ' + EM + EN + ' ' + 'f</p>\n'
    parser = commonmark.Parser(options=dict(smart=True))
    ast = parser.parse(md)
    renderer = commonmark.HtmlRenderer()
    html = renderer.render(ast)
    self.assertEqual(html, expected_html)