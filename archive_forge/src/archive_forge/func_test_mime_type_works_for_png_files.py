from pecan.middleware.static import (StaticFileMiddleware, FileWrapper,
from pecan.tests import PecanTestCase
import os
def test_mime_type_works_for_png_files(self):
    result = self._request('/static_fixtures/self.png')
    assert self._get_response_header('Content-Type') == 'image/png'
    result.close()