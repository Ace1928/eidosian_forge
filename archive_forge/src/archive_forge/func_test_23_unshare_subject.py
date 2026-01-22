import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_23_unshare_subject():
    target_project = central.select.project('pyxnat_tests2')
    shared_subj_1 = target_project.subject(_id_set1['sid'])
    assert shared_subj_1.exists()
    assert subj_1.shares().get() == ['pyxnat_tests', 'pyxnat_tests2']
    subj_1.unshare('pyxnat_tests2')
    shared_subj_1 = target_project.subject(_id_set1['sid'])
    assert not shared_subj_1.exists()
    assert subj_1.shares().get() == ['pyxnat_tests']