from novaclient.tests.functional import base
def test_os_service_disable_enable(self):
    for serv in self.client.services.list():
        if serv.binary != 'nova-compute':
            continue
        host = self._get_column_value_from_single_row_table(self.nova('service-list --binary %s' % serv.binary), 'Host')
        service = self.nova('service-disable %s' % host)
        self.addCleanup(self.nova, 'service-enable', params=host)
        status = self._get_column_value_from_single_row_table(service, 'Status')
        self.assertEqual('disabled', status)
        service = self.nova('service-enable %s' % host)
        status = self._get_column_value_from_single_row_table(service, 'Status')
        self.assertEqual('enabled', status)