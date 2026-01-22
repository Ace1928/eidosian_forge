from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_all_set_exclusive(self):
    self.run_bzr_error(['Only one option can be set.'], ['config', '--all', 'hello=world'])