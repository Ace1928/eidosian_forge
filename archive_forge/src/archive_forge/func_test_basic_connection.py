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
def test_basic_connection(self):
    self.store.configure()
    connection = self.store.get_connection(self.location, context=self.context)
    self.assertIsNone(connection.authurl)
    self.assertEqual('1', connection.auth_version)
    self.assertIsNone(connection.user)
    self.assertIsNone(connection.tenant_name)
    self.assertIsNone(connection.key)
    self.assertEqual('https://example.com', connection.preauthurl)
    self.assertEqual('0123', connection.preauthtoken)
    self.assertEqual({}, connection.os_options)