import copy
import os
import sys
from unittest import mock
import testtools
from osc_lib import shell
from osc_lib.tests import utils
@testtools.skip('skip until bug 1444983 is resolved')
def test_help_options(self):
    flag = '-h list server'
    kwargs = {'deferred_help': True}
    with mock.patch(self.app_patch + '.initialize_app', self.app):
        _shell, _cmd = (utils.make_shell(), flag)
        utils.fake_execute(_shell, _cmd)
        self.assertEqual(kwargs['deferred_help'], _shell.options.deferred_help)