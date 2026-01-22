from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_cmd_line(self):
    self.breezy_config.set_user_option('hello', 'world')
    script.run_script(self, '            $ brz config -Ohello=bzr\n            cmdline:\n              hello = bzr\n            breezy:\n              [DEFAULT]\n              hello = world\n            ')