from unittest import mock
from oslo_config import cfg
from oslotest import base
from cinderclient import exceptions as cinder_exception
from glance_store.common import attachment_state_manager as attach_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
@mock.patch.object(cinder_utils.API, 'attachment_create')
def test_attach_fails(self, mock_attach_create):
    mock_attach_create.side_effect = cinder_exception.BadRequest(code=400)
    self.assertRaises(cinder_exception.BadRequest, self.m.attach, mock.sentinel.client, mock.sentinel.volume_id, mock.sentinel.host, mode=mock.sentinel.mode)