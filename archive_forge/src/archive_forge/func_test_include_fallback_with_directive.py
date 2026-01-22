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
def test_include_fallback_with_directive(self):
    dirname = tempfile.mkdtemp(suffix='genshi_test')
    try:
        file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
        try:
            file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude"\n                      xmlns:py="http://genshi.edgewall.org/">\n                  <xi:include href="tmpl1.html"><xi:fallback>\n                    <py:if test="True">tmpl1.html not found</py:if>\n                  </xi:fallback></xi:include>\n                </html>')
        finally:
            file2.close()
        loader = TemplateLoader([dirname])
        tmpl = loader.load('tmpl2.html')
        self.assertEqual('<html>\n                    tmpl1.html not found\n                </html>', tmpl.generate(debug=True).render(encoding=None))
    finally:
        shutil.rmtree(dirname)