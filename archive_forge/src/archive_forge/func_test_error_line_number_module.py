import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_error_line_number_module(self):
    loader = None

    def load_generate(path, **kwargs):
        assert loader is not None
        return loader.load(path).generate(**kwargs)
    loader = DictLoader({'base.html': "{% module Template('sub.html') %}", 'sub.html': '{{1/0}}'}, namespace={'_tt_modules': ObjectDict(Template=load_generate)})
    try:
        loader.load('base.html').generate()
        self.fail('did not get expected exception')
    except ZeroDivisionError:
        exc_stack = traceback.format_exc()
        self.assertTrue('# base.html:1' in exc_stack)
        self.assertTrue('# sub.html:1' in exc_stack)