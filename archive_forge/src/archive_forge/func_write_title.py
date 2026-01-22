from __future__ import unicode_literals
import argparse
import io
import logging
import os
import sys
import textwrap
import cmakelang
from cmakelang.lint import lintdb
from tangent.tooling.gendoc import format_directive
def write_title(outfile, title, rulerchar=None, numrule=1):
    if rulerchar is None:
        rulerchar = '-'
    if numrule == 2:
        outfile.write(rulerchar * len(title))
        outfile.write('\n')
    outfile.write(title)
    outfile.write('\n')
    outfile.write(rulerchar * len(title))
    outfile.write('\n\n')