import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_if_empty_body(self):
    template = Template(utf8('{% if True %}{% else %}{% end %}'))
    self.assertEqual(template.generate(), b'')