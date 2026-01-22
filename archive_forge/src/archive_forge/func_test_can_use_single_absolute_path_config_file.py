import os
from keystone.server.flask import core as server_flask
from keystone.tests import unit
def test_can_use_single_absolute_path_config_file(self):
    cfg = self.custom_config_files[0]
    cfgpath = os.path.join(self.custom_config_dir, cfg)
    env = {'OS_KEYSTONE_CONFIG_FILES': cfgpath}
    config_files = server_flask._get_config_files(env)
    self.assertListEqual(config_files, [cfgpath])