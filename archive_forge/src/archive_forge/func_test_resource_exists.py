import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_resource_exists():
    assert resource_1.exists()
    assert isinstance(resource_1, object)
    assert isinstance(resource_1, pyxnat.core.resources.Resource)
    assert str(resource_1) != '<Resource Object> IOP'