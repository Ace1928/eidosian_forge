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
def get_stylesheet(self):
    """Get the stylesheet from the visitor.
        Ask the visitor to setup the page.
        """
    s1 = self.visitor.setup_page()
    return s1