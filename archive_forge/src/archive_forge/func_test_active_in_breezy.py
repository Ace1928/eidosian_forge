from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_active_in_breezy(self):
    script.run_script(self, '            $ brz config -d tree --scope breezy file\n            breezy\n            ')