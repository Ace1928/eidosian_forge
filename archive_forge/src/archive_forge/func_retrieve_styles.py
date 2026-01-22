import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def retrieve_styles(self, extension):
    """Retrieve the stylesheet from either a .xml file or from
        a .odt (zip) file.  Return the content as a string.
        """
    s2 = None
    stylespath = self.settings.stylesheet
    ext = os.path.splitext(stylespath)[1]
    if ext == '.xml':
        stylesfile = open(stylespath, 'r')
        s1 = stylesfile.read()
        stylesfile.close()
    elif ext == extension:
        zfile = zipfile.ZipFile(stylespath, 'r')
        s1 = zfile.read('styles.xml')
        s2 = zfile.read('content.xml')
        zfile.close()
    else:
        raise RuntimeError('stylesheet path (%s) must be %s or .xml file' % (stylespath, extension))
    self.str_stylesheet = s1
    self.str_stylesheetcontent = s2
    self.dom_stylesheet = etree.fromstring(self.str_stylesheet)
    self.dom_stylesheetcontent = etree.fromstring(self.str_stylesheetcontent)
    self.table_styles = self.extract_table_styles(s2)