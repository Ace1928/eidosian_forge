import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def get_edge_preproc_code(self, edge, attribute='texlbl'):
    lblstyle = edge.attr.get('lblstyle', '')
    text = edge.attr.get(attribute, '')
    if lblstyle:
        return '  \\tikz \\node[%s] {%s};\n' % (lblstyle, text)
    else:
        return '\\tikz \\node ' + '{' + text + '};'