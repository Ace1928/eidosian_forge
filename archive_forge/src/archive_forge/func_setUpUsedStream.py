import unittest
from io import BytesIO, StringIO
from testtools.compat import _b
import subunit
def setUpUsedStream(self):
    self.input_stream.write(_b('tags: global\ntest passed\nsuccess passed\ntest failed\ntags: local\nfailure failed\ntest error\nerror error\ntest skipped\nskip skipped\ntest todo\nxfail todo\n'))
    self.input_stream.seek(0)
    self.test.run(self.result)