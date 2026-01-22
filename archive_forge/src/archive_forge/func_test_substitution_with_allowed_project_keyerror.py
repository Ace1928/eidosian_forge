import uuid
from keystone.common import utils
from keystone import exception
from keystone.tests import unit
def test_substitution_with_allowed_project_keyerror(self):
    url_template = 'http://server:9090/$(project_id)s/$(user_id)s'
    values = {'user_id': 'B'}
    self.assertIsNone(utils.format_url(url_template, values, silent_keyerror_failures=['project_id']))