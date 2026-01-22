import datetime
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import filtering
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def test_list_filtered_domains(self):
    """GET /domains?enabled=0.

        Test Plan:

        - Update policy for no protection on api
        - Filter by the 'enabled' boolean to get disabled domains, which
          should return just domainC
        - Try the filter using different ways of specifying True/False
          to test that our handling of booleans in filter matching is
          correct

        """
    new_policy = {'identity:list_domains': []}
    self._set_policy(new_policy)
    r = self.get('/domains?enabled=0', auth=self.auth)
    id_list = self._get_id_list_from_ref_list(r.result.get('domains'))
    self.assertEqual(1, len(id_list))
    self.assertIn(self.domainC['id'], id_list)
    for val in ('0', 'false', 'False', 'FALSE', 'n', 'no', 'off'):
        r = self.get('/domains?enabled=%s' % val, auth=self.auth)
        id_list = self._get_id_list_from_ref_list(r.result.get('domains'))
        self.assertEqual([self.domainC['id']], id_list)
    for val in ('1', 'true', 'True', 'TRUE', 'y', 'yes', 'on'):
        r = self.get('/domains?enabled=%s' % val, auth=self.auth)
        id_list = self._get_id_list_from_ref_list(r.result.get('domains'))
        self.assertEqual(3, len(id_list))
        self.assertIn(self.domainA['id'], id_list)
        self.assertIn(self.domainB['id'], id_list)
        self.assertIn(CONF.identity.default_domain_id, id_list)
    r = self.get('/domains?enabled', auth=self.auth)
    id_list = self._get_id_list_from_ref_list(r.result.get('domains'))
    self.assertEqual(3, len(id_list))
    self.assertIn(self.domainA['id'], id_list)
    self.assertIn(self.domainB['id'], id_list)
    self.assertIn(CONF.identity.default_domain_id, id_list)