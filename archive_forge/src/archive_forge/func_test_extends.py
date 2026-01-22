import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_extends(self):
    loader = DictLoader({'base.html': '<title>{% block title %}default title{% end %}</title>\n<body>{% block body %}default body{% end %}</body>\n', 'page.html': '{% extends "base.html" %}\n{% block title %}page title{% end %}\n{% block body %}page body{% end %}\n'})
    self.assertEqual(loader.load('page.html').generate(), b'<title>page title</title>\n<body>page body</body>\n')