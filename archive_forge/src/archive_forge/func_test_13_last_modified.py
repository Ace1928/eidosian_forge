import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_13_last_modified():
    sid = subj_1.id()
    t1 = central.select('/project/pyxnat_tests').last_modified()[sid]
    subj_1.attrs.set('age', '26')
    assert subj_1.attrs.get('age') == '26'
    t2 = central.select('/project/pyxnat_tests').last_modified()[sid]
    assert t1 != t2