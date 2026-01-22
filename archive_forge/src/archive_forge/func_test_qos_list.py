from cinderclient.tests.functional import base
def test_qos_list(self):
    qos_list = self.cinder('qos-list')
    self.assertTableHeaders(qos_list, ['ID', 'Name', 'Consumer', 'specs'])