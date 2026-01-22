import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
def test_request_and_yield_offsets(self):
    data = b'abcdefghijklmnopqrstuvwxyz'
    self.checkRequestAndYield([(0, b'a'), (5, b'f'), (10, b'klm')], data, [(0, 1), (5, 1), (10, 3)])
    self.checkRequestAndYield([(0, b'a'), (1, b'b'), (10, b'klm')], data, [(0, 1), (1, 1), (10, 3)])
    self.checkRequestAndYield([(0, b'a'), (10, b'k'), (4, b'efg'), (1, b'bcd')], data, [(0, 1), (10, 1), (4, 3), (1, 3)])