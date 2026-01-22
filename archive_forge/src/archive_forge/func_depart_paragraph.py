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
def depart_paragraph(self, node):
    self.in_paragraph = False
    self.set_to_parent()
    if self.in_header:
        self.header_content.append(self.current_element.getchildren()[-1])
        self.current_element.remove(self.current_element.getchildren()[-1])
    elif self.in_footer:
        self.footer_content.append(self.current_element.getchildren()[-1])
        self.current_element.remove(self.current_element.getchildren()[-1])