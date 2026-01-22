import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api import base as api_base
from mistralclient.api.v2 import actions
from mistralclient.tests.unit.v2 import base
def test_get_with_namespace(self):
    self.requests_mock.get(self.TEST_URL + URL_TEMPLATE_NAME % 'action/namespace', json=ACTION)
    action = self.actions.get('action', 'namespace')
    self.assertIsNotNone(action)
    self.assertEqual(actions.Action(self.actions, ACTION).to_dict(), action.to_dict())