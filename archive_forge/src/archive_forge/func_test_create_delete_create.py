import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_create_delete_create():
    from uuid import uuid1
    sid = uuid1().hex
    s = proj_1.subject(sid)
    s.create()
    assert s.exists()
    s.delete()
    s.create()
    s.delete()
    assert not s.exists()