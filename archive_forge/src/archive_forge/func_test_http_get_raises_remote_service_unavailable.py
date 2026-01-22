from unittest import mock
import requests
import glance_store
from glance_store._drivers import http
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
def test_http_get_raises_remote_service_unavailable(self):
    """Test http store raises RemoteServiceUnavailable."""
    uri = 'http://netloc/path/to/file.tar.gz'
    loc = location.get_location_from_uri(uri, conf=self.conf)
    self.assertRaises(exceptions.RemoteServiceUnavailable, self.store.get, loc)