import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_list_role_assignments_filters(self):
    domain_data = self._get_domain_data()
    user_data = self._get_user_data(domain_id=domain_data.domain_id)
    role_data = self._get_role_data()
    response = [{'links': 'https://example.com', 'role': {'id': role_data.role_id}, 'scope': {'domain': {'id': domain_data.domain_id}}, 'user': {'id': user_data.user_id}}]
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='role_assignments', qs_elements=['scope.domain.id=%s' % domain_data.domain_id, 'user.id=%s' % user_data.user_id, 'effective=True']), status_code=200, json={'role_assignments': response}, complete_qs=True)])
    params = dict(user=user_data.user_id, domain=domain_data.domain_id, effective=True)
    ret = self.cloud.list_role_assignments(filters=params)
    self.assertThat(len(ret), matchers.Equals(1))
    self.assertThat(ret[0].user['id'], matchers.Equals(user_data.user_id))
    self.assertThat(ret[0].role['id'], matchers.Equals(role_data.role_id))
    self.assertThat(ret[0].scope['domain']['id'], matchers.Equals(domain_data.domain_id))