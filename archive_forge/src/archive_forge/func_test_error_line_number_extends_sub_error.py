import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_error_line_number_extends_sub_error(self):
    loader = DictLoader({'base.html': "{% block 'block' %}{% end %}", 'sub.html': "\n{% extends 'base.html' %}\n{% block 'block' %}\n{{1/0}}\n{% end %}\n            "})
    try:
        loader.load('sub.html').generate()
        self.fail('did not get expected exception')
    except ZeroDivisionError:
        self.assertTrue('# sub.html:4 (via base.html:1)' in traceback.format_exc())