import os
from uuid import uuid1
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_01_set_param():
    scan.set_param('foo', 'foostring')
    scan.set_param('bar', '1')
    assert scan.params() == ['foo', 'bar']