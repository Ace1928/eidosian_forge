import os
from keystone.server.flask import core as server_flask
from keystone.tests import unit
def test_config_files_have_default_values_with_empty_envars(self):
    env = {'OS_KEYSTONE_CONFIG_FILES': '', 'OS_KEYSTONE_CONFIG_DIR': ''}
    config_files = server_flask._get_config_files(env)
    config_files.sort()
    expected_config_files = []
    self.assertListEqual(config_files, expected_config_files)