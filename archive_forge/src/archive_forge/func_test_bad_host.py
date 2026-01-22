import errno
import os
import signal
import socket
from subprocess import Popen
import sys
import time
import unittest
from tornado.netutil import (
from tornado.testing import AsyncTestCase, gen_test, bind_unused_port
from tornado.test.util import skipIfNoNetwork
import typing
@gen_test
def test_bad_host(self: typing.Any):
    with self.assertRaises(IOError):
        yield self.resolver.resolve('an invalid domain', 80, socket.AF_UNSPEC)