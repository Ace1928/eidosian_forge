from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_remove_no_option(self):
    self.run_bzr_error(['--remove expects an option to remove.'], ['config', '--remove'])