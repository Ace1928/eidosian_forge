import uuid
from keystone.common import utils
from keystone import exception
from keystone.tests import unit
def test_raises_malformed_on_incomplete_format(self):
    self.assertRaises(exception.MalformedEndpoint, utils.format_url, 'http://server:9090/$(tenant_id)', {'tenant_id': 'A'})