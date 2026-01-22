import doctest
import os
import shutil
import tempfile
import unittest
from genshi.template.base import TemplateSyntaxError
from genshi.template.loader import TemplateLoader
from genshi.template.text import OldTextTemplate, NewTextTemplate
def test_end_with_args(self):
    tmpl = NewTextTemplate("\n{% if foo %}\n  bar\n{% end 'if foo' %}")
    self.assertEqual('\n', tmpl.generate(foo=False).render(encoding=None))