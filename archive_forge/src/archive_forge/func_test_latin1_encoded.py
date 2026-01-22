import doctest
import os
import shutil
import tempfile
import unittest
from genshi.template.base import TemplateSyntaxError
from genshi.template.loader import TemplateLoader
from genshi.template.text import OldTextTemplate, NewTextTemplate
def test_latin1_encoded(self):
    text = u'$fooö$bar'.encode('iso-8859-1')
    tmpl = NewTextTemplate(text, encoding='iso-8859-1')
    self.assertEqual(u'xöy', tmpl.generate(foo='x', bar='y').render(encoding=None))