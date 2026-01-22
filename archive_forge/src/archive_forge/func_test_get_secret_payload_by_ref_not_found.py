import collections
from unittest import mock
from barbicanclient import exceptions
from heat.common import exception
from heat.engine.clients.os import barbican
from heat.tests import common
from heat.tests import utils
def test_get_secret_payload_by_ref_not_found(self):
    exc = exceptions.HTTPClientError(message='Not Found', status_code=404)
    self.barbican_client.secrets.get.side_effect = exc
    self.assertRaises(exception.EntityNotFound, self.barbican_plugin.get_secret_payload_by_ref, 'secret')