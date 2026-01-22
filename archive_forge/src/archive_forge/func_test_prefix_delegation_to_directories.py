import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def test_prefix_delegation_to_directories(self):
    """
        Test prefix delegation with the following layout:
        
        templates/foo.html
        sub1/templates/tmpl1.html
        sub2/templates/tmpl2.html
        
        Where sub1 and sub2 are prefixes, and both tmpl1.html and tmpl2.html
        incldue foo.html.
        """
    dir1 = os.path.join(self.dirname, 'templates')
    os.mkdir(dir1)
    file1 = open(os.path.join(dir1, 'foo.html'), 'w')
    try:
        file1.write('<div>Included foo</div>')
    finally:
        file1.close()
    dir2 = os.path.join(self.dirname, 'sub1', 'templates')
    os.makedirs(dir2)
    file2 = open(os.path.join(dir2, 'tmpl1.html'), 'w')
    try:
        file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n              <xi:include href="../foo.html" /> from sub1\n            </html>')
    finally:
        file2.close()
    dir3 = os.path.join(self.dirname, 'sub2', 'templates')
    os.makedirs(dir3)
    file3 = open(os.path.join(dir3, 'tmpl2.html'), 'w')
    try:
        file3.write('<div>tmpl2</div>')
    finally:
        file3.close()
    loader = TemplateLoader([dir1, TemplateLoader.prefixed(sub1=dir2, sub2=dir3)])
    tmpl = loader.load('sub1/tmpl1.html')
    self.assertEqual('<html>\n              <div>Included foo</div> from sub1\n            </html>', tmpl.generate().render(encoding=None))