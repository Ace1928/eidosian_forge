import os
import stat
import sys
from io import BytesIO
from .. import errors, osutils, pyutils, tests
from .. import transport as _mod_transport
from .. import urlutils
from ..errors import ConnectionError, PathError, TransportNotPossible
from ..osutils import getcwd
from ..transport import (ConnectedTransport, FileExists, NoSuchFile, Transport,
from ..transport.memory import MemoryTransport
from ..transport.remote import RemoteTransport
from . import TestNotApplicable, TestSkipped, multiply_tests, test_server
from .test_transport import TestTransportImplementation
def test_readv_with_adjust_for_latency_with_big_file(self):
    transport = self.get_transport()
    if transport.is_readonly():
        with open('a', 'w') as f:
            f.write('a' * 1024 * 1024)
    else:
        transport.put_bytes('a', b'a' * 1024 * 1024)
    broken_vector = [(465219, 800), (225221, 800), (445548, 800), (225037, 800), (221357, 800), (437077, 800), (947670, 800), (465373, 800), (947422, 800)]
    results = list(transport.readv('a', broken_vector, True, 1024 * 1024))
    found_items = [False] * 9
    for pos, (start, length) in enumerate(broken_vector):
        for offset, data in results:
            if offset <= start and start + length <= offset + len(data):
                found_items[pos] = True
    self.assertEqual([True] * 9, found_items)