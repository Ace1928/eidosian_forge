from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_breezy_config(self):
    self.breezy_config.set_user_option('hello', 'world')
    script.run_script(self, '            $ brz config -d tree\n            breezy:\n              [DEFAULT]\n              hello = world\n            ')