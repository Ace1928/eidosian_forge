import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
def test_get_os_instance_usage_audit_log(self):
    begin, end = self._get_begin_end_time()
    expected = {'hosts_not_run': '[]', 'log': '{}', 'num_hosts': '0', 'num_hosts_done': '0', 'num_hosts_not_run': '0', 'num_hosts_running': '0', 'overall_status': 'ALL hosts done. 0 errors.', 'total_errors': '0', 'total_instances': '0', 'period_beginning': str(begin), 'period_ending': str(end)}
    output = self.nova('instance-usage-audit-log')
    for key in expected.keys():
        self.assertEqual(expected[key], self._get_value_from_the_table(output, key))