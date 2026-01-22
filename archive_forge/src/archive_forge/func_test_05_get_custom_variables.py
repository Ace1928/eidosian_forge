import os
from uuid import uuid1
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_05_get_custom_variables():
    project.get_custom_variables()