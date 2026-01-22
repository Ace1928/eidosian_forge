import os
import os.path as op
import tempfile 
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
def test_save_config_current_dir():
    f = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False)
    cfg = op.basename(f.name)
    try:
        f.close()
        central.save_config(cfg)
        assert op.isfile(cfg)
    finally:
        os.remove(cfg)
    assert not op.isfile(cfg)