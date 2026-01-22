import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_disable_replication_group(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.38'))
    expected = {'disable_replication': {}}
    g0 = cs.groups.list()[0]
    grp = g0.disable_replication()
    self._assert_request_id(grp)
    cs.assert_called('POST', '/groups/1234/action', body=expected)
    grp = cs.groups.disable_replication('1234')
    self._assert_request_id(grp)
    cs.assert_called('POST', '/groups/1234/action', body=expected)
    grp = cs.groups.disable_replication(g0)
    self._assert_request_id(grp)
    cs.assert_called('POST', '/groups/1234/action', body=expected)