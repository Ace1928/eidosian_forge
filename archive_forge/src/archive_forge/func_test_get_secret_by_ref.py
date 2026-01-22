import collections
from unittest import mock
from barbicanclient import exceptions
from heat.common import exception
from heat.engine.clients.os import barbican
from heat.tests import common
from heat.tests import utils
def test_get_secret_by_ref(self):
    secret = collections.namedtuple('Secret', ['name'])('foo')
    self.barbican_client.secrets.get.return_value = secret
    self.assertEqual(secret, self.barbican_plugin.get_secret_by_ref('secret'))