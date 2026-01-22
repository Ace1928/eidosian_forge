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
def to_latex_length(self, length_str, pxunit=None):
    """Convert `length_str` with rst lenght to LaTeX length
        """
    if pxunit is not None:
        sys.stderr.write('deprecation warning: LaTeXTranslator.to_latex_length() option `pxunit` will be removed.')
    match = re.match('(\\d*\\.?\\d*)\\s*(\\S*)', length_str)
    if not match:
        return length_str
    value, unit = match.groups()[:2]
    if unit in ('', 'pt'):
        length_str = '%sbp' % value
    elif unit == '%':
        length_str = '%.3f\\linewidth' % (float(value) / 100.0)
    elif self.is_xetex and unit == 'px':
        self.fallbacks['_providelength'] = PreambleCmds.providelength
        self.fallbacks['px'] = '\n\\DUprovidelength{\\pdfpxdimen}{1bp}\n'
        length_str = '%s\\pdfpxdimen' % value
    return length_str