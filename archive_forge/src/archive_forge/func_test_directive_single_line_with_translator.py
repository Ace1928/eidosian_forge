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
def test_directive_single_line_with_translator(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n            <py:for each="i in range(2)"><py:for each="j in range(1)">\n                <span py:content="i + j"></span>\n            </py:for></py:for>\n        </div>')
    translator = Translator(lambda s: s)
    tmpl.add_directives(Translator.NAMESPACE, translator)
    self.assertEqual('<div>\n                <span>0</span>\n                <span>1</span>\n        </div>', str(tmpl.generate()))