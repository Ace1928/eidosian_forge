import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_error_line_number_expression(self):
    loader = DictLoader({'test.html': 'one\ntwo{{1/0}}\nthree\n        '})
    try:
        loader.load('test.html').generate()
        self.fail('did not get expected exception')
    except ZeroDivisionError:
        self.assertTrue('# test.html:2' in traceback.format_exc())