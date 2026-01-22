import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def init_template_vars(self):
    DotConvBase.init_template_vars(self)
    if self.options.get('crop'):
        cropcode = '\\usepackage[active,tightpage]{preview}\n' + '\\PreviewEnvironment{tikzpicture}\n' + '\\setlength\\PreviewBorder{%s}' % self.options.get('margin', '0pt')
    else:
        cropcode = ''
    variables = {'<<cropcode>>': cropcode}
    self.templatevars.update(variables)