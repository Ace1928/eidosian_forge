import collections
import copy
import os.path
from oslo_serialization import jsonutils
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api import base as api_base
from mistralclient.api.v2 import environments
from mistralclient.tests.unit.v2 import base
from mistralclient import utils
def test_create_with_json_file_uri(self):
    path = pkg.resource_filename('mistralclient', 'tests/unit/resources/env_v2.json')
    path = os.path.abspath(path)
    uri = parse.urljoin('file:', request.pathname2url(path))
    data = collections.OrderedDict(utils.load_content(utils.get_contents_if_file(uri)))
    self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, status_code=201, json=data)
    file_input = {'file': uri}
    env = self.environments.create(**file_input)
    self.assertIsNotNone(env)
    expected_data = copy.deepcopy(data)
    expected_data['variables'] = jsonutils.dumps(expected_data['variables'])
    self.assertEqual(expected_data, self.requests_mock.last_request.json())