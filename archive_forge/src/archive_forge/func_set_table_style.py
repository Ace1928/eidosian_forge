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
def set_table_style(self, table_style, classes):
    borders = [cls.replace('nolines', 'borderless') for cls in table_style + classes if cls in ('standard', 'booktabs', 'borderless', 'nolines')]
    try:
        self.borders = borders[-1]
    except IndexError:
        self.borders = 'standard'
    self.colwidths_auto = 'colwidths-auto' in classes and 'colwidths-given' not in table_style or ('colwidths-auto' in table_style and 'colwidths-given' not in classes)