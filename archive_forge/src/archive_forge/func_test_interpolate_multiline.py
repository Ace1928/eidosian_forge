import doctest
import os
import pickle
import shutil
import sys
import tempfile
import unittest
import six
from genshi.compat import BytesIO, StringIO
from genshi.core import Markup
from genshi.filters.i18n import Translator
from genshi.input import XML
from genshi.template.base import BadDirectiveError, TemplateSyntaxError
from genshi.template.loader import TemplateLoader, TemplateNotFound
from genshi.template.markup import MarkupTemplate
def test_interpolate_multiline(self):
    tmpl = MarkupTemplate("<root>${dict(\n          bar = 'baz'\n        )[foo]}</root>")
    self.assertEqual('<root>baz</root>', str(tmpl.generate(foo='bar')))