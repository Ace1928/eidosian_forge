import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def test_abspath_include_caching_without_search_path(self):
    file1 = open(os.path.join(self.dirname, 'tmpl1.html'), 'w')
    try:
        file1.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n              <xi:include href="tmpl2.html" />\n            </html>')
    finally:
        file1.close()
    file2 = open(os.path.join(self.dirname, 'tmpl2.html'), 'w')
    try:
        file2.write('<div>Included</div>')
    finally:
        file2.close()
    os.mkdir(os.path.join(self.dirname, 'sub'))
    file3 = open(os.path.join(self.dirname, 'sub', 'tmpl1.html'), 'w')
    try:
        file3.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n              <xi:include href="tmpl2.html" />\n            </html>')
    finally:
        file3.close()
    file4 = open(os.path.join(self.dirname, 'sub', 'tmpl2.html'), 'w')
    try:
        file4.write('<div>Included from sub</div>')
    finally:
        file4.close()
    loader = TemplateLoader()
    tmpl1 = loader.load(os.path.join(self.dirname, 'tmpl1.html'))
    self.assertEqual('<html>\n              <div>Included</div>\n            </html>', tmpl1.generate().render(encoding=None))
    tmpl2 = loader.load(os.path.join(self.dirname, 'sub', 'tmpl1.html'))
    self.assertEqual('<html>\n              <div>Included from sub</div>\n            </html>', tmpl2.generate().render(encoding=None))
    assert 'tmpl2.html' not in loader._cache