from __future__ import print_function, unicode_literals, with_statement
import re
import sys
from os import path
from shutil import rmtree
from subprocess import PIPE, Popen
from tempfile import mkdtemp
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.output import bibtex
from pybtex.errors import report_error
from pybtex.exceptions import PybtexError
def write_aux(filename, citations):
    with open(filename, 'w') as aux_file:
        for citation in citations:
            aux_file.write('\\citation{%s}\n' % citation)
        aux_file.write('\\bibdata{test}\n')
        aux_file.write('\\bibstyle{test}\n')