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
def find_first_text_p(self, el):
    """Search the generated doc and return the first <text:p> element.
        """
    if el.tag == 'text:p' or el.tag == 'text:h':
        return el
    elif el.getchildren():
        for child in el.getchildren():
            el1 = self.find_first_text_p(child)
            if el1 is not None:
                return el1
        return None
    else:
        return None