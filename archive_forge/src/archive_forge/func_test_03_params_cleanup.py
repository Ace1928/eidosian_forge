import os
from uuid import uuid1
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_03_params_cleanup():
    project.subject(sid).delete()
    assert not project.subject(sid).exists()