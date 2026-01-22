from heat.engine import support
from heat.tests import common
def test_previous_status(self):
    sstatus = support.SupportStatus(status=support.DEPRECATED, version='5.0.0', previous_status=support.SupportStatus(status=support.SUPPORTED, version='2015.1'))
    self.assertEqual(support.DEPRECATED, sstatus.status)
    self.assertEqual('5.0.0', sstatus.version)
    self.assertEqual(support.SUPPORTED, sstatus.previous_status.status)
    self.assertEqual('2015.1', sstatus.previous_status.version)
    self.assertEqual({'status': 'DEPRECATED', 'version': '5.0.0', 'message': None, 'previous_status': {'status': 'SUPPORTED', 'version': '2015.1', 'message': None, 'previous_status': None}}, sstatus.to_dict())