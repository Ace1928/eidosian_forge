from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_prearchive_get():
    from pyxnat.core import manage
    pa = manage.PreArchive(central)
    pa.get()