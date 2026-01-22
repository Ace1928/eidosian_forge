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
def test_connection_closed_reporting(self):
    requester, response_handler = self.make_client_protocol()
    requester.call(b'hello')
    ex = self.assertRaises(errors.ConnectionReset, response_handler.read_response_tuple)
    self.assertEqual('Connection closed: Unexpected end of message. Please check connectivity and permissions, and report a bug if problems persist. ', str(ex))