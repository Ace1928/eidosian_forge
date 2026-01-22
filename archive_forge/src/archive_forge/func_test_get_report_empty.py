from unittest import mock
from osprofiler.drivers.mongodb import MongoDB
from osprofiler.tests import test
def test_get_report_empty(self):
    self.mongodb.db = mock.MagicMock()
    self.mongodb.db.profiler.find.return_value = []
    expected = {'info': {'name': 'total', 'started': 0, 'finished': None, 'last_trace_started': None}, 'children': [], 'stats': {}}
    base_id = '10'
    self.assertEqual(expected, self.mongodb.get_report(base_id))