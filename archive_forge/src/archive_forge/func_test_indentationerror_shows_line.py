import io
import os.path
import platform
import re
import sys
import traceback
import unittest
from textwrap import dedent
from tempfile import TemporaryDirectory
from IPython.core.ultratb import ColorTB, VerboseTB
from IPython.testing import tools as tt
from IPython.testing.decorators import onlyif_unicode_paths, skip_without
from IPython.utils.syspathcontext import prepended_to_syspath
import sys
def test_indentationerror_shows_line(self):
    with tt.AssertPrints('IndentationError'):
        with tt.AssertPrints('zoon()', suppress=False):
            ip.run_cell(indentationerror_file)
    with TemporaryDirectory() as td:
        fname = os.path.join(td, 'foo.py')
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(indentationerror_file)
        with tt.AssertPrints('IndentationError'):
            with tt.AssertPrints('zoon()', suppress=False):
                ip.magic('run %s' % fname)