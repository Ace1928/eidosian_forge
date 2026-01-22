import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_default_on(self):
    loader = DictLoader(self.templates, autoescape='xhtml_escape')
    name = 'Bobby <table>s'
    self.assertEqual(loader.load('escaped.html').generate(name=name), b'Bobby &lt;table&gt;s')
    self.assertEqual(loader.load('unescaped.html').generate(name=name), b'Bobby <table>s')
    self.assertEqual(loader.load('default.html').generate(name=name), b'Bobby &lt;table&gt;s')
    self.assertEqual(loader.load('include.html').generate(name=name), b'escaped: Bobby &lt;table&gt;s\nunescaped: Bobby <table>s\ndefault: Bobby &lt;table&gt;s\n')