from unittest import mock
import uuid
from cinderclient import exceptions as cinder_exc
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients.os import cinder
from heat.tests import common
from heat.tests import utils
def test_cinder_api_v3(self):
    ctx = utils.dummy_context()
    self.patchobject(ctx.keystone_session, 'get_endpoint')
    client = ctx.clients.client('cinder')
    self.assertEqual('3.0', client.version)