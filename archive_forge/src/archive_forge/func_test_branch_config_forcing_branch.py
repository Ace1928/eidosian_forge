from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_branch_config_forcing_branch(self):
    script.run_script(self, '            $ brz config -d tree --scope branch --remove file\n            $ brz config -d tree --all file\n            locations:\n              [.../work/tree]\n              file = locations\n            breezy:\n              [DEFAULT]\n              file = breezy\n            ')
    script.run_script(self, '            $ brz config -d tree --scope locations --remove file\n            $ brz config -d tree --all file\n            breezy:\n              [DEFAULT]\n              file = breezy\n            ')