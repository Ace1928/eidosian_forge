import copy
import os
import sys
from unittest import mock
import testtools
from osc_lib import shell
from osc_lib.tests import utils
def test_shell_args_cert_options(self):
    """Test client cert options"""
    _shell = utils.make_shell()
    utils.fake_execute(_shell, 'module list')
    self.assertEqual('', _shell.options.cert)
    self.assertEqual('', _shell.options.key)
    self.assertIsNone(_shell.client_manager.cert)
    utils.fake_execute(_shell, '--os-cert mycert module list')
    self.assertEqual('mycert', _shell.options.cert)
    self.assertEqual('', _shell.options.key)
    self.assertEqual('mycert', _shell.client_manager.cert)
    utils.fake_execute(_shell, '--os-key mickey module list')
    self.assertEqual('', _shell.options.cert)
    self.assertEqual('mickey', _shell.options.key)
    self.assertIsNone(_shell.client_manager.cert)
    utils.fake_execute(_shell, '--os-cert mycert --os-key mickey module list')
    self.assertEqual('mycert', _shell.options.cert)
    self.assertEqual('mickey', _shell.options.key)
    self.assertEqual(('mycert', 'mickey'), _shell.client_manager.cert)