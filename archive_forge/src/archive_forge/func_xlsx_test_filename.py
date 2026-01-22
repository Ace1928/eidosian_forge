from __future__ import absolute_import, print_function, division
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xlsx import fromxlsx, toxlsx, appendxlsx
from petl.test.helpers import ieq, eq_
@pytest.fixture()
def xlsx_test_filename():
    pkg_resources = pytest.importorskip('pkg_resources')
    return pkg_resources.resource_filename('petl', 'test/resources/test.xlsx')