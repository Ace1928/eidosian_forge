import copy
from unittest import mock
from oslo_serialization import jsonutils
from oslo_policy import shell
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_flatten_from_file(self):
    target = {'target': {'secret': {'project_id': '1234'}}}
    self.create_config_file('target.json', jsonutils.dumps(target))
    with open(self.get_config_file_fullname('target.json'), 'r') as fh:
        target_from_file = fh.read()
    result = shell.flatten(jsonutils.loads(target_from_file))
    self.assertEqual(result, {'target.secret.project_id': '1234'})