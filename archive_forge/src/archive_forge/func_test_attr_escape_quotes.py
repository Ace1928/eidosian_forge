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
def test_attr_escape_quotes(self):
    """
        Verify that outputting context data in attribtes escapes quotes.
        """
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <elem class="$myvar"/>\n        </div>')
    self.assertEqual('<div>\n          <elem class="&#34;foo&#34;"/>\n        </div>', str(tmpl.generate(myvar='"foo"')))