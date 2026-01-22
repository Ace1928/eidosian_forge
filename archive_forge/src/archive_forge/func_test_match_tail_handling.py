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
def test_match_tail_handling(self):
    xml = '<rhyme xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="*[@type]">\n            ${select(\'.\')}\n          </py:match>\n\n          <lines>\n            <first type="one">fish</first>\n            <second type="two">fish</second>\n            <third type="red">fish</third>\n            <fourth type="blue">fish</fourth>\n          </lines>\n        </rhyme>'
    tmpl = MarkupTemplate(xml, filename='test.html')
    self.assertEqual('<rhyme>\n          <lines>\n            <first type="one">fish</first>\n            <second type="two">fish</second>\n            <third type="red">fish</third>\n            <fourth type="blue">fish</fourth>\n          </lines>\n        </rhyme>', tmpl.generate().render(encoding=None))