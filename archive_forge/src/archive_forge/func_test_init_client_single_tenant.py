import copy
from unittest import mock
import fixtures
import hashlib
import http.client
import importlib
import io
import tempfile
import uuid
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests_mock
import swiftclient
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
import glance_store.multi_backend as store
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
@mock.patch('glance_store._drivers.swift.store.ks_identity')
@mock.patch('glance_store._drivers.swift.store.ks_session')
@mock.patch('glance_store._drivers.swift.store.ks_client')
def test_init_client_single_tenant(self, mock_client, mock_session, mock_identity):
    """Test that keystone client was initialized correctly"""
    store = Store(self.conf, backend='swift1')
    store.configure()
    uri = 'swift://%s:key@auth_address/glance/%s' % (self.swift_store_user, FAKE_UUID)
    loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
    ctxt = mock.MagicMock()
    store.init_client(location=loc.store_location, context=ctxt)
    mock_identity.V3Password.assert_called_once_with(auth_url=loc.store_location.swift_url + '/', username='user1', password='key', project_name='tenant', project_domain_id='default', project_domain_name=None, user_domain_id='default', user_domain_name=None)
    mock_session.Session.assert_called_once_with(auth=mock_identity.V3Password(), verify=True)
    mock_client.Client.assert_called_once_with(session=mock_session.Session())