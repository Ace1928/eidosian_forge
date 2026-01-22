import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
from keystone.tests.unit import utils as test_utils
def test_list_projects_filtering_multiple_any_tag_filters(self):
    """Call ``GET /projects?tags-any={tags}&not-tags-any={tags}``."""
    project1, tags1 = self._create_project_and_tags()
    project2, tags2 = self._create_project_and_tags(num_of_tags=2)
    url = '/projects?tags-any=%(value1)s&not-tags-any=%(value2)s'
    resp = self.get(url % {'value1': tags1[0], 'value2': tags2[0]})
    self.assertValidProjectListResponse(resp)
    pids = [p['id'] for p in resp.result['projects']]
    self.assertIn(project1['id'], pids)
    self.assertNotIn(project2['id'], pids)