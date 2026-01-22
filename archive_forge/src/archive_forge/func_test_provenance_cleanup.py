from uuid import uuid1
from pyxnat.tests import skip_if_no_network
from pyxnat import Interface
import os.path as op
@skip_if_no_network
def test_provenance_cleanup():
    project.subject(sid).delete()
    assert not project.subject(sid).exists()