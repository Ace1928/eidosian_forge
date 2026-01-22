from pecan.middleware.static import (StaticFileMiddleware, FileWrapper,
from pecan.tests import PecanTestCase
import os
def test_no_file_found_causes_passthrough(self):
    result = self._request('/static_fixtures/nosuchfile.txt')
    assert not isinstance(result, FileWrapper)
    assert result == ['Hello world!\n']