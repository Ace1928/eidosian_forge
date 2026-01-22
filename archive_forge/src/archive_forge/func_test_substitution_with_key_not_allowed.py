import uuid
from keystone.common import utils
from keystone import exception
from keystone.tests import unit
def test_substitution_with_key_not_allowed(self):
    url_template = 'http://server:9090/$(project_id)s/$(user_id)s/$(admin_token)s'
    values = {'user_id': 'B', 'admin_token': 'C'}
    self.assertRaises(exception.MalformedEndpoint, utils.format_url, url_template, values)