import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_extended_block(self):
    loader = DictLoader(self.templates)

    def render(name):
        return loader.load(name).generate(name='<script>')
    self.assertEqual(render('escaped_extends_unescaped.html'), b'base: <script>')
    self.assertEqual(render('escaped_overrides_unescaped.html'), b'extended: &lt;script&gt;')
    self.assertEqual(render('unescaped_extends_escaped.html'), b'base: &lt;script&gt;')
    self.assertEqual(render('unescaped_overrides_escaped.html'), b'extended: <script>')