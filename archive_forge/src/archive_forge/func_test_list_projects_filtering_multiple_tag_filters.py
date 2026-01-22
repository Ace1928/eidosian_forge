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
def test_list_projects_filtering_multiple_tag_filters(self):
    """Call ``GET /projects?tags={tags}&tags-any={tags}``."""
    project1, tags1 = self._create_project_and_tags(num_of_tags=2)
    project2, tags2 = self._create_project_and_tags(num_of_tags=2)
    project3, tags3 = self._create_project_and_tags(num_of_tags=2)
    tags1_query = ','.join(tags1)
    resp = self.patch('/projects/%(project_id)s' % {'project_id': project3['id']}, body={'project': {'tags': tags1}})
    tags1.append(tags2[0])
    resp = self.patch('/projects/%(project_id)s' % {'project_id': project1['id']}, body={'project': {'tags': tags1}})
    url = '/projects?tags=%(value1)s&tags-any=%(value2)s'
    resp = self.get(url % {'value1': tags1_query, 'value2': ','.join(tags2)})
    self.assertValidProjectListResponse(resp)
    self.assertEqual(len(resp.result['projects']), 1)
    self.assertIn(project1['id'], resp.result['projects'][0]['id'])