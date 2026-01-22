import http.client as http
import uuid
from oslo_serialization import jsonutils
import requests
from glance.tests import functional
def test_task_not_allowed_non_admin(self):
    self.start_servers(**self.__dict__.copy())
    roles = {'X-Roles': 'member'}
    path = self._url('/v2/tasks')
    response = requests.get(path, headers=self._headers(roles))
    self.assertEqual(http.FORBIDDEN, response.status_code)