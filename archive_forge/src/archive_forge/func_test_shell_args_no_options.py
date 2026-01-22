import copy
import os
import sys
from unittest import mock
import testtools
from osc_lib import shell
from osc_lib.tests import utils
def test_shell_args_no_options(self):
    _shell = utils.make_shell()
    with mock.patch('osc_lib.shell.OpenStackShell.initialize_app', self.app):
        utils.fake_execute(_shell, 'list user')
        self.app.assert_called_with(['list', 'user'])