import json
from datetime import datetime
from unittest import TestCase
import macaroonbakery.bakery as bakery
import pymacaroons
from macaroonbakery._utils import cookie
from pymacaroons.serializers import json_serializer
def test_cookie_with_hostname_not_fqdn(self):
    c = cookie('http://myhost', 'test', 'value')
    self.assertEqual(c.domain, 'myhost.local')