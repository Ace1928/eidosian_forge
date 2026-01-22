import os
from keystone.server.flask import core as server_flask
from keystone.tests import unit
def test_can_use_default_config_files_with_custom_config_dir(self):
    env = {'OS_KEYSTONE_CONFIG_DIR': self.custom_config_dir}
    config_files = server_flask._get_config_files(env)
    config_files.sort()
    expected_config_files = [os.path.join(self.custom_config_dir, self.default_config_file)]
    self.assertListEqual(config_files, expected_config_files)