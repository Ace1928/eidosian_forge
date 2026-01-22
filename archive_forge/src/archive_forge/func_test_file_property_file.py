import re
import unittest
from wsme import exc
from wsme import types
def test_file_property_file(self):

    class buffer:

        def read(self):
            return 'from-file'
    buf = buffer()
    f = types.File(file=buf)
    assert f.file is buf