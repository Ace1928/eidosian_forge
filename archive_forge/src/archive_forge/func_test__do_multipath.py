import builtins
import errno
import os.path
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import executor
from os_brick.initiator.connectors import nvmeof
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick.tests.initiator import test_connector
from os_brick import utils
@ddt.data({'result': False, 'use_multipath': False, 'ana_support': True}, {'result': False, 'use_multipath': False, 'ana_support': False}, {'result': False, 'use_multipath': True, 'ana_support': False}, {'result': True, 'use_multipath': True, 'ana_support': True})
@ddt.unpack
def test__do_multipath(self, result, use_multipath, ana_support):
    self.connector.use_multipath = use_multipath
    self.connector.native_multipath_supported = ana_support
    self.assertIs(result, self.connector._do_multipath())