import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
def run_isolated(klass, self, result):
    """Run a test suite or case in a subprocess, using the run method on klass.
    """
    c2pread, c2pwrite = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(c2pread)
        os.dup2(c2pwrite, 1)
        os.close(c2pwrite)
        stream = os.fdopen(1, 'wb')
        result = TestProtocolClient(stream)
        klass.run(self, result)
        stream.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        os.close(c2pwrite)
        protocol = TestProtocolServer(result)
        fileobj = os.fdopen(c2pread, 'rb')
        protocol.readFrom(fileobj)
        os.waitpid(pid, 0)
    return result