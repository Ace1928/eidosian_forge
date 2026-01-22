import re
import unittest
from wsme import exc
from wsme import types
def test_file_content_overrides_file(self):

    class buffer:

        def read(self):
            return 'from-file'
    f = types.File(content='from-content', file=buffer())
    assert f.content == 'from-content'