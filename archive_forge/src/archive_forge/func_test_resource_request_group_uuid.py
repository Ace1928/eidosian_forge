from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_resource_request_group_uuid(self):
    try:
        place_utils.resource_request_group_uuid(namespace=self._uuid_ns, qos_rules=[mock.MagicMock(id='fake_id_0'), mock.MagicMock(id='fake_id_1')])
    except Exception:
        self.fail('could not generate resource request group uuid')