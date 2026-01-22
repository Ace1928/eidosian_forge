from octavia_lib.api.drivers import exceptions
from octavia_lib.tests.unit import base
def test_NotImplementedError(self):
    not_implemented_error = exceptions.NotImplementedError(user_fault_string=self.user_fault_string, operator_fault_string=self.operator_fault_string)
    self.assertEqual(self.user_fault_string, not_implemented_error.user_fault_string)
    self.assertEqual(self.operator_fault_string, not_implemented_error.operator_fault_string)
    self.assertIsInstance(not_implemented_error, Exception)