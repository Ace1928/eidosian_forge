import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_unicode_apply(self):

    def upper(s):
        return to_unicode(s).upper()
    template = Template(utf8('{% apply upper %}foo é{% end %}'))
    self.assertEqual(template.generate(upper=upper), utf8('FOO É'))