import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_error_line_number_extends_base_error(self):
    loader = DictLoader({'base.html': '{{1/0}}', 'sub.html': "{% extends 'base.html' %}"})
    try:
        loader.load('sub.html').generate()
        self.fail('did not get expected exception')
    except ZeroDivisionError:
        exc_stack = traceback.format_exc()
    self.assertTrue('# base.html:1' in exc_stack)