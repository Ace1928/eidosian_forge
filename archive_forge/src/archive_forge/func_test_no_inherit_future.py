import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
@unittest.skip('no testable future imports')
def test_no_inherit_future(self):
    self.assertEqual(1 / 2, 0.5)
    template = Template('{{ 1 / 2 }}')
    self.assertEqual(template.generate(), '0')