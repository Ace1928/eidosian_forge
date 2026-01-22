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
@onlyif_unicode_paths
def test_nonascii_path(self):
    with TemporaryDirectory(suffix=u'é') as td:
        fname = os.path.join(td, u'fooé.py')
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(file_1)
        with prepended_to_syspath(td):
            ip.run_cell('import foo')
        with tt.AssertPrints('ZeroDivisionError'):
            ip.run_cell('foo.f()')