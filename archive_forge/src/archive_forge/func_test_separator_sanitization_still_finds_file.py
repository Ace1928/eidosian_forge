from pecan.middleware.static import (StaticFileMiddleware, FileWrapper,
from pecan.tests import PecanTestCase
import os
def test_separator_sanitization_still_finds_file(self):
    os.altsep = ':'
    result = self._request(':static_fixtures:text.txt')
    assert isinstance(result, FileWrapper)
    result.close()