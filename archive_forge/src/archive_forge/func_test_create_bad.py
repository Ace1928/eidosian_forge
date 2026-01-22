import tempfile
from pecan.templating import RendererFactory, format_line_context
from pecan.tests import PecanTestCase
def test_create_bad(self):
    self.assertEqual(self.rf.get('doesnotexist', '/'), None)