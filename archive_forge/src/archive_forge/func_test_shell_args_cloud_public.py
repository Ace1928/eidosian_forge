import copy
import os
import sys
from unittest import mock
import testtools
from osc_lib import shell
from osc_lib.tests import utils
@mock.patch('openstack.config.loader.OpenStackConfig._load_vendor_file')
@mock.patch('openstack.config.loader.OpenStackConfig._load_config_file')
def test_shell_args_cloud_public(self, config_mock, public_mock):
    """Test cloud config options with the vendor file"""
    config_mock.return_value = ('file.yaml', copy.deepcopy(CLOUD_2))
    public_mock.return_value = ('file.yaml', copy.deepcopy(PUBLIC_1))
    _shell = utils.make_shell()
    utils.fake_execute(_shell, '--os-cloud megacloud module list')
    self.assertEqual('megacloud', _shell.cloud.name)
    self.assertEqual(DEFAULT_AUTH_URL, _shell.cloud.config['auth']['auth_url'])
    self.assertEqual('cake', _shell.cloud.config['donut'])
    self.assertEqual('heart-o-gold', _shell.cloud.config['auth']['project_name'])
    self.assertEqual('zaphod', _shell.cloud.config['auth']['username'])
    self.assertEqual('occ-cloud', _shell.cloud.config['region_name'])
    self.assertEqual('occ-cloud', _shell.client_manager.region_name)
    self.assertEqual('mycert', _shell.cloud.config['cert'])
    self.assertEqual('mickey', _shell.cloud.config['key'])
    self.assertEqual(('mycert', 'mickey'), _shell.client_manager.cert)