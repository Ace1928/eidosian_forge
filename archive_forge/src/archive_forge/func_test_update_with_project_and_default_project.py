from unittest import mock
import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import users
def test_update_with_project_and_default_project(self, ref=None):
    self.deprecations.expect_deprecations()
    ref = self.new_ref()
    req_ref = ref.copy()
    req_ref.pop('id')
    param_ref = req_ref.copy()
    self.stub_entity('PATCH', [self.collection_key, ref['id']], status_code=200, entity=ref)
    param_ref['project_id'] = 'project'
    params = utils.parameterize(param_ref)
    returned = self.manager.update(ref['id'], **params)
    self.assertIsInstance(returned, self.model)
    for attr in ref:
        self.assertEqual(getattr(returned, attr), ref[attr], 'Expected different %s' % attr)
    self.assertEntityRequestBodyIs(req_ref)