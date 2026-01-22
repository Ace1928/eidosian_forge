import uuid
from keystone.common import utils
from keystone import exception
from keystone.tests import unit
def test_successful_formatting(self):
    url_template = 'http://server:9090/$(tenant_id)s/$(user_id)s/$(project_id)s'
    project_id = uuid.uuid4().hex
    values = {'tenant_id': 'A', 'user_id': 'B', 'project_id': project_id}
    actual_url = utils.format_url(url_template, values)
    expected_url = 'http://server:9090/A/B/%s' % (project_id,)
    self.assertEqual(expected_url, actual_url)