from unittest import mock
import uuid
from cinderclient import exceptions as cinder_exc
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients.os import cinder
from heat.tests import common
from heat.tests import utils
def test_cinder_api_not_supported(self):
    ctx = utils.dummy_context()
    self.patchobject(ctx.keystone_session, 'get_endpoint', side_effect=[ks_exceptions.EndpointNotFound, ks_exceptions.EndpointNotFound])
    self.assertRaises(exception.Error, ctx.clients.client, 'cinder')