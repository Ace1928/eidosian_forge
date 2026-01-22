import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def test_relative_include_samesubdir(self):
    file1 = open(os.path.join(self.dirname, 'tmpl1.html'), 'w')
    try:
        file1.write('<div>Included tmpl1.html</div>')
    finally:
        file1.close()
    os.mkdir(os.path.join(self.dirname, 'sub'))
    file2 = open(os.path.join(self.dirname, 'sub', 'tmpl1.html'), 'w')
    try:
        file2.write('<div>Included sub/tmpl1.html</div>')
    finally:
        file2.close()
    file3 = open(os.path.join(self.dirname, 'sub', 'tmpl2.html'), 'w')
    try:
        file3.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n              <xi:include href="tmpl1.html" />\n            </html>')
    finally:
        file3.close()
    loader = TemplateLoader([self.dirname])
    tmpl = loader.load('sub/tmpl2.html')
    self.assertEqual('<html>\n              <div>Included sub/tmpl1.html</div>\n            </html>', tmpl.generate().render(encoding=None))