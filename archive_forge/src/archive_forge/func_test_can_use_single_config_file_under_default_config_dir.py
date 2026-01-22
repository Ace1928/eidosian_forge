import os
from keystone.server.flask import core as server_flask
from keystone.tests import unit
def test_can_use_single_config_file_under_default_config_dir(self):
    cfg = self.custom_config_files[0]
    env = {'OS_KEYSTONE_CONFIG_FILES': cfg}
    config_files = server_flask._get_config_files(env)
    expected_config_files = [cfg]
    self.assertListEqual(config_files, expected_config_files)