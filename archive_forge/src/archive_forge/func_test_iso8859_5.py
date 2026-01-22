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
def test_iso8859_5(self):
    with TemporaryDirectory() as td:
        fname = os.path.join(td, 'dfghjkl.py')
        with io.open(fname, 'w', encoding='iso-8859-5') as f:
            f.write(iso_8859_5_file)
        with prepended_to_syspath(td):
            ip.run_cell('from dfghjkl import fail')
        with tt.AssertPrints('ZeroDivisionError'):
            with tt.AssertPrints(u'дбИЖ', suppress=False):
                ip.run_cell('fail()')