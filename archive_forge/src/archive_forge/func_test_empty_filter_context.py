from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
def test_empty_filter_context(self):
    ctx = ContentFilterContext()
    self.assertEqual(None, ctx.relpath())