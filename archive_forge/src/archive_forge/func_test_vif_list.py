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
@mock.patch.object(node.NodeManager, '_list', autospec=True)
def test_vif_list(self, _list_mock):
    kwargs = {'node_ident': NODE1['uuid']}
    final_path = '/v1/nodes/%s/vifs' % NODE1['uuid']
    self.mgr.vif_list(**kwargs)
    _list_mock.assert_called_once_with(mock.ANY, final_path, 'vifs', os_ironic_api_version=None, global_request_id=None)