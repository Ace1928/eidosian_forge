import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_io_textwrapper(self):
    text_io = io.TextIOWrapper(io.BytesIO())
    self.assertThat(unicode_output_stream(text_io), Is(text_io))
    unicode_output_stream(text_io).write('foo')