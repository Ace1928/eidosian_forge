import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def test_node_set_secure_boot_on(self):
    secure_boot = self.mgr.set_secure_boot(NODE1['uuid'], 'on')
    body = {'target': True}
    expect = [('PUT', '/v1/nodes/%s/states/secure_boot' % NODE1['uuid'], {}, body)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(secure_boot)