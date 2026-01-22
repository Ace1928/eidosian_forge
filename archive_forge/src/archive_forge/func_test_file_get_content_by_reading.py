import re
import unittest
from wsme import exc
from wsme import types
def test_file_get_content_by_reading(self):

    class buffer:

        def read(self):
            return 'abcdef'
    f = types.File(file=buffer())
    assert f.content == 'abcdef'