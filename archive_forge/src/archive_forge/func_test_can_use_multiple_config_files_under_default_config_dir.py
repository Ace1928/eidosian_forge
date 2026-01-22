import os
from keystone.server.flask import core as server_flask
from keystone.tests import unit
def test_can_use_multiple_config_files_under_default_config_dir(self):
    env = {'OS_KEYSTONE_CONFIG_FILES': ';'.join(self.custom_config_files)}
    config_files = server_flask._get_config_files(env)
    config_files.sort()
    expected_config_files = self.custom_config_files
    self.assertListEqual(config_files, expected_config_files)
    config_with_empty_strings = self.custom_config_files + ['', ' ']
    env = {'OS_KEYSTONE_CONFIG_FILES': ';'.join(config_with_empty_strings)}
    config_files = server_flask._get_config_files(env)
    config_files.sort()
    self.assertListEqual(config_files, expected_config_files)