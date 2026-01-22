import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_utf8_in_file(self):
    tmpl = self.loader.load('utf8.html')
    result = tmpl.generate()
    self.assertEqual(to_unicode(result).strip(), 'Héllo')