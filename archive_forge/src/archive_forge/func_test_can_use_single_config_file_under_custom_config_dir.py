import os
from keystone.server.flask import core as server_flask
from keystone.tests import unit
def test_can_use_single_config_file_under_custom_config_dir(self):
    cfg = self.custom_config_files[0]
    env = {'OS_KEYSTONE_CONFIG_DIR': self.custom_config_dir, 'OS_KEYSTONE_CONFIG_FILES': cfg}
    config_files = server_flask._get_config_files(env)
    config_files.sort()
    expected_config_files = [os.path.join(self.custom_config_dir, cfg)]
    self.assertListEqual(config_files, expected_config_files)