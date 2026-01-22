import os
import os.path as op
import tempfile 
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
def test_save_config():
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as tempdir:
        cfg = op.join(tempdir, '.xnat.cfg')
        central.save_config(cfg)
        assert op.isfile(cfg)
    assert not op.isfile(cfg)