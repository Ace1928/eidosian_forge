import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_15_getitem_slice():
    projects = central.select.projects()
    assert projects.first().id() == next(projects[:1]).id()
    piter = projects.__iter__()
    next(piter)
    next(piter)
    next(piter)
    for pobj in projects[3:6]:
        assert next(piter).id() == pobj.id()