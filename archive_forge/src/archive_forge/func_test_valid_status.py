from heat.engine import support
from heat.tests import common
def test_valid_status(self):
    for sstatus in support.SUPPORT_STATUSES:
        previous = support.SupportStatus(version='test_version')
        status = support.SupportStatus(status=sstatus, message='test_message', version='test_version', previous_status=previous)
        self.assertEqual(sstatus, status.status)
        self.assertEqual('test_message', status.message)
        self.assertEqual('test_version', status.version)
        self.assertEqual(previous, status.previous_status)
        self.assertEqual({'status': sstatus, 'message': 'test_message', 'version': 'test_version', 'previous_status': {'status': 'SUPPORTED', 'message': None, 'version': 'test_version', 'previous_status': None}}, status.to_dict())