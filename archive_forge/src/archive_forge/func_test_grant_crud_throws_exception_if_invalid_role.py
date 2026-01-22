from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_grant_crud_throws_exception_if_invalid_role(self):
    """Ensure RoleNotFound thrown if role does not exist."""

    def assert_role_not_found_exception(f, **kwargs):
        self.assertRaises(exception.RoleNotFound, f, role_id=uuid.uuid4().hex, **kwargs)
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user_resp = PROVIDERS.identity_api.create_user(user)
    group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
    group_resp = PROVIDERS.identity_api.create_group(group)
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project_resp = PROVIDERS.resource_api.create_project(project['id'], project)
    for manager_call in [PROVIDERS.assignment_api.create_grant, PROVIDERS.assignment_api.get_grant]:
        assert_role_not_found_exception(manager_call, user_id=user_resp['id'], project_id=project_resp['id'])
        assert_role_not_found_exception(manager_call, group_id=group_resp['id'], project_id=project_resp['id'])
        assert_role_not_found_exception(manager_call, user_id=user_resp['id'], domain_id=CONF.identity.default_domain_id)
        assert_role_not_found_exception(manager_call, group_id=group_resp['id'], domain_id=CONF.identity.default_domain_id)
    assert_role_not_found_exception(PROVIDERS.assignment_api.delete_grant, user_id=user_resp['id'], project_id=project_resp['id'])
    assert_role_not_found_exception(PROVIDERS.assignment_api.delete_grant, group_id=group_resp['id'], project_id=project_resp['id'])
    assert_role_not_found_exception(PROVIDERS.assignment_api.delete_grant, user_id=user_resp['id'], domain_id=CONF.identity.default_domain_id)
    assert_role_not_found_exception(PROVIDERS.assignment_api.delete_grant, group_id=group_resp['id'], domain_id=CONF.identity.default_domain_id)