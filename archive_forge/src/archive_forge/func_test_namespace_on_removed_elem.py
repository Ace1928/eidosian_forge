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
def test_namespace_on_removed_elem(self):
    """
        Verify that a namespace declaration on an element that is removed from
        the generated stream does not get pushed up to the next non-stripped
        element (see ticket #107).
        """
    tmpl = MarkupTemplate('<?xml version="1.0"?>\n        <Test xmlns:py="http://genshi.edgewall.org/">\n          <Size py:if="0" xmlns:t="test">Size</Size>\n          <Item/>\n        </Test>')
    self.assertEqual('<?xml version="1.0"?>\n<Test>\n          \n          <Item/>\n        </Test>', str(tmpl.generate()))