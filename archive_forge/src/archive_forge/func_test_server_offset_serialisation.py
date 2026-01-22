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
def test_server_offset_serialisation(self):
    """The Smart protocol serialises offsets as a comma and 
 string.

        We check a number of boundary cases are as expected: empty, one offset,
        one with the order of reads not increasing (an out of order read), and
        one that should coalesce.
        """
    requester, response_handler = self.make_client_protocol()
    self.assertOffsetSerialisation([], b'', requester)
    self.assertOffsetSerialisation([(1, 2)], b'1,2', requester)
    self.assertOffsetSerialisation([(10, 40), (0, 5)], b'10,40\n0,5', requester)
    self.assertOffsetSerialisation([(1, 2), (3, 4), (100, 200)], b'1,2\n3,4\n100,200', requester)