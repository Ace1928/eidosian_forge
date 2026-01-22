import hashlib
from unittest import mock
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import uuidutils
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib import constants
from neutron_lib import exceptions
from neutron_lib.plugins import utils
from neutron_lib.tests import _base as base
def test_verify_tunnel_range_known_tunnel_type(self):
    mock_fns = [mock.Mock(return_value=False) for _ in range(3)]
    mock_map = {constants.TYPE_GRE: mock_fns[0], constants.TYPE_VXLAN: mock_fns[1], constants.TYPE_GENEVE: mock_fns[2]}
    with mock.patch.dict(utils._TUNNEL_MAPPINGS, mock_map):
        for t in [constants.TYPE_GRE, constants.TYPE_VXLAN, constants.TYPE_GENEVE]:
            self.assertRaises(exceptions.NetworkTunnelRangeError, utils.verify_tunnel_range, [0, 1], t)
        for f in mock_fns:
            f.assert_called_once_with(0)