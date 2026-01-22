import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_route_code(self):
    self.check_event(CONSTANT_ROUTE_CODE, 'success', route_code='source', test_id='bar')