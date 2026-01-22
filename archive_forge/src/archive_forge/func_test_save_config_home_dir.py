import os
import os.path as op
import tempfile 
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
def test_save_config_home_dir():
    with tempfile.TemporaryDirectory(dir=op.expanduser('~')) as tempdir:
        cfg = op.join('~', op.basename(tempdir), '.xnat.cfg')
        central.save_config(cfg)
        assert op.isfile(op.expanduser(cfg))
    assert not op.isfile(op.expanduser(cfg))