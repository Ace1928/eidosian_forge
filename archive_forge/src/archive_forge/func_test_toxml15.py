from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def test_toxml15():
    _check_toxml(_TABLE1, _TAB_ABZ, check=('row', 'col'), root='table', rows='row/col', prologue=_TAG_A0, epilogue=_TAG_Z9)