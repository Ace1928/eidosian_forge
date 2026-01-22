from .. import utils
import builtins
import io
import logging
import os
import pytest
import tempfile
import sys
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import callbacks
from rpy2.rinterface_lib import openrlib
def test_showfiles_default(capsys):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b'abc')
        tmp.close()
        filenames = (tmp,)
        headers = ('',)
        wtitle = ''
        pager = ''
        captured = capsys.readouterr()
        callbacks.showfiles(tuple((x.name for x in filenames)), headers, wtitle, pager)
        captured.out.endswith('---')
        os.unlink(tmp.name)