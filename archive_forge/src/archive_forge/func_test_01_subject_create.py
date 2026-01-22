import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_01_subject_create():
    assert not subj_1.exists()
    subj_1.create()
    assert subj_1.exists()