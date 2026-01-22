import doctest
import os
import shutil
import tempfile
import unittest
from genshi.template.base import TemplateSyntaxError
from genshi.template.loader import TemplateLoader
from genshi.template.text import OldTextTemplate, NewTextTemplate
def test_comment(self):
    tmpl = NewTextTemplate('{# a comment #}')
    self.assertEqual('', tmpl.generate().render(encoding=None))