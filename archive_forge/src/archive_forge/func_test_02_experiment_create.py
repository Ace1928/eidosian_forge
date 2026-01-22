import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_02_experiment_create():
    assert not expe_1.exists()
    expe_1.create()
    assert expe_1.exists()
    expe_1.trigger(fix_types=True, pipelines=True, scan_headers=False)