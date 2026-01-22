import uuid
import fixtures
from unittest import mock
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import utils
def test_nomemcache(self):
    conf = {'admin_token': 'admin_token1', 'auth_host': 'keystone.example.com', 'auth_port': '1234', 'memcached_servers': ','.join(MEMCACHED_SERVERS), 'www_authenticate_uri': 'https://keystone.example.com:1234'}
    self.create_simple_middleware(conf=conf)