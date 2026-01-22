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
def minitoc(self, node, title, depth):
    """Generate a local table of contents with LaTeX package minitoc"""
    section_name = self.d_class.section(self.section_level)
    minitoc_names = {'part': 'part', 'chapter': 'mini'}
    if 'chapter' not in self.d_class.sections:
        minitoc_names['section'] = 'sect'
    try:
        minitoc_name = minitoc_names[section_name]
    except KeyError:
        self.warn('Skipping local ToC at %s level.\n' % section_name + '  Feature not supported with option "use-latex-toc"', base_node=node)
        return
    self.requirements['minitoc'] = PreambleCmds.minitoc
    self.requirements['minitoc-' + minitoc_name] = '\\do%stoc' % minitoc_name
    maxdepth = len(self.d_class.sections)
    self.requirements['minitoc-%s-depth' % minitoc_name] = '\\mtcsetdepth{%stoc}{%d}' % (minitoc_name, maxdepth)
    offset = {'sect': 1, 'mini': 0, 'part': 0}
    if 'chapter' in self.d_class.sections:
        offset['part'] = -1
    if depth:
        self.out.append('\\setcounter{%stocdepth}{%d}' % (minitoc_name, depth + offset[minitoc_name]))
    self.out.append('\\mtcsettitle{%stoc}{%s}\n' % (minitoc_name, title))
    self.out.append('\\%stoc\n' % minitoc_name)