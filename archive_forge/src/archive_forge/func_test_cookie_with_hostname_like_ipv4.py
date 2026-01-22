import json
from datetime import datetime
from unittest import TestCase
import macaroonbakery.bakery as bakery
import pymacaroons
from macaroonbakery._utils import cookie
from pymacaroons.serializers import json_serializer
def test_cookie_with_hostname_like_ipv4(self):
    c = cookie('http://1.2.3.4.com', 'test', 'value')
    self.assertEqual(c.domain, '1.2.3.4.com')