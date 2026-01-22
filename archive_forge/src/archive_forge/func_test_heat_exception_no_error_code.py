from heat.common import exception
from heat.tests import common
def test_heat_exception_no_error_code(self):
    ex = TestException.SampleExceptionNoErorCode()
    self.assertEqual('Test exception', ex.message)