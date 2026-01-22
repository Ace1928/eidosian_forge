from octavia_lib.api.drivers import exceptions
from octavia_lib.tests.unit import base
def test_NotFound(self):
    not_found_error = exceptions.NotFound(user_fault_string=self.user_fault_string, operator_fault_string=self.operator_fault_string)
    self.assertEqual(self.user_fault_string, not_found_error.user_fault_string)
    self.assertEqual(self.operator_fault_string, not_found_error.operator_fault_string)
    self.assertIsInstance(not_found_error, Exception)