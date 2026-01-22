from heat.common import exception
from heat.tests import common
def test_heat_exception_with_error_code(self):
    ex = TestException.SampleException()
    self.assertEqual('HEAT-E100 Test exception', ex.message)