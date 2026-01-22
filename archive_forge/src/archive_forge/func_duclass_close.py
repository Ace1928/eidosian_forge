import sys
import os
import time
import re
import string
import urllib.request, urllib.parse, urllib.error
from docutils import frontend, nodes, languages, writers, utils, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import pick_math_environment, unichar2tex
def duclass_close(self, node):
    """Close a group of class declarations."""
    for cls in reversed(node['classes']):
        if cls.startswith('language-'):
            language = self.babel.language_name(cls[9:])
            if language:
                self.babel.otherlanguages[language] = True
                self.out.append('\\end{selectlanguage}\n')
        else:
            self.fallbacks['DUclass'] = PreambleCmds.duclass
            self.out.append('\\end{DUclass}\n')