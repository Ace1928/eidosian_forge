import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_12_file_last_modified():
    f = subj_1.resource('test').file('hello.txt')
    assert isinstance(f.last_modified(), str)
    assert len(f.last_modified()) > 0
    f.delete()
    assert not f.exists()
    r = subj_1.resource('test')
    r.delete()
    assert not r.exists()