from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def test_toxml14():
    table1 = [['foo', 'bar'], ['a', 1], ['b', 2]]
    _check_toxml(table1, table1, style='attribute', rows='row/col')
    _check_toxml(table1, table1, style='name', rows='row/col')