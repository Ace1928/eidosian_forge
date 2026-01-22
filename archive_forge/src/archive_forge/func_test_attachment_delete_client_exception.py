from unittest import mock
import uuid
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from oslo_config import cfg
from oslotest import base
from glance_store.common import cinder_utils
def test_attachment_delete_client_exception(self):
    self.fake_client.attachments.delete.side_effect = cinder_exception.ClientException(code=1)
    self.assertRaises(cinder_exception.ClientException, self.volume_api.attachment_delete, self.fake_client, self.fake_attach_id)