import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_break_continue(self):
    template = Template(utf8('{% for i in range(10) %}\n    {% if i == 2 %}\n        {% continue %}\n    {% end %}\n    {{ i }}\n    {% if i == 6 %}\n        {% break %}\n    {% end %}\n{% end %}'))
    result = template.generate()
    result = b''.join(result.split())
    self.assertEqual(result, b'013456')