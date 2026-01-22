import logging
import os
import pipes
import platform
import sys
import tempfile
import time
from unittest import mock
import testtools
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep import priv_context
from oslo_privsep.tests import testctx
def test_helper_command_default_dirtoo(self):
    self.privsep_conf.config_file = ['/bar.conf', '/baz.conf']
    self.privsep_conf.config_dir = ['/foo.d']
    _, temp_path = tempfile.mkstemp()
    cmd = testctx.context.helper_command(temp_path)
    expected = ['sudo', 'privsep-helper', '--config-file', '/bar.conf', '--config-file', '/baz.conf', '--config-dir', '/foo.d', '--privsep_context', testctx.context.pypath, '--privsep_sock_path', temp_path]
    self.assertEqual(expected, cmd)