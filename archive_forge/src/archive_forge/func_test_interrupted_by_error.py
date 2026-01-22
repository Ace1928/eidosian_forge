import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
def test_interrupted_by_error(self):
    response_handler = self.make_response_handler(interrupted_body_stream)
    stream = response_handler.read_streamed_body()
    self.assertEqual(b'aaa', next(stream))
    self.assertEqual(b'bbb', next(stream))
    exc = self.assertRaises(errors.ErrorFromSmartServer, next, stream)
    self.assertEqual((b'error', b'Exception', b'Boom!'), exc.error_tuple)