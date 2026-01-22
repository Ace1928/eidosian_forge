import testtools
from magnumclient.tests import utils
from magnumclient.v1 import stats
def test_stats_with_project_id(self):
    expect = [('GET', '/v1/stats?project_id=%s' % CLUSTER2['project_id'], {}, None)]
    stats = self.mgr.list(project_id=CLUSTER2['project_id'])
    self.assertEqual(expect, self.api.calls)
    expected_stats = {'clusters': 1, 'nodes': C2[nc] + C2[mc]}
    self.assertEqual(expected_stats, stats._info)