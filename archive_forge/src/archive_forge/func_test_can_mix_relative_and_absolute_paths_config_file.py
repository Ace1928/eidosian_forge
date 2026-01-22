import os
from keystone.server.flask import core as server_flask
from keystone.tests import unit
def test_can_mix_relative_and_absolute_paths_config_file(self):
    cfg0 = self.custom_config_files[0]
    cfgpath0 = os.path.join(self.custom_config_dir, self.custom_config_files[0])
    cfgpath1 = os.path.join(self.custom_config_dir, self.custom_config_files[1])
    env = {'OS_KEYSTONE_CONFIG_DIR': self.custom_config_dir, 'OS_KEYSTONE_CONFIG_FILES': ';'.join([cfg0, cfgpath1])}
    config_files = server_flask._get_config_files(env)
    config_files.sort()
    expected_config_files = [cfgpath0, cfgpath1]
    expected_config_files.sort()
    self.assertListEqual(config_files, expected_config_files)
    env = {'OS_KEYSTONE_CONFIG_FILES': ';'.join([cfg0, cfgpath1])}
    config_files = server_flask._get_config_files(env)
    config_files.sort()
    expected_config_files = [cfg0, cfgpath1]
    expected_config_files.sort()
    self.assertListEqual(config_files, expected_config_files)