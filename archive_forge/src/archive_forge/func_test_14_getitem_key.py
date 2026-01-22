import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_14_getitem_key():
    projects = central.select.projects()
    assert projects.first().id() == projects[0].id()
    piter = projects.__iter__()
    next(piter)
    assert next(piter).id() == projects[1].id()