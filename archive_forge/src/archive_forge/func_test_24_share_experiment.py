import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_24_share_experiment():
    target_project = central.select.project('pyxnat_tests2')
    shared_expe_1 = target_project.experiment(_id_set1['eid'])
    assert not shared_expe_1.exists()
    assert expe_1.shares().get() == ['pyxnat_tests']
    expe_1.share('pyxnat_tests2')
    shared_expe_1 = target_project.experiment(_id_set1['eid'])
    assert shared_expe_1.exists()
    assert expe_1.shares().get() == ['pyxnat_tests', 'pyxnat_tests2']