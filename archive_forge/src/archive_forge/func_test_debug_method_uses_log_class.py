import os
import sys
import zlib
from io import StringIO
from unittest import mock
import requests_mock
import libcloud
from libcloud.http import LibcloudConnection
from libcloud.test import unittest
from libcloud.common.base import Connection
from libcloud.utils.loggingconnection import LoggingConnection
def test_debug_method_uses_log_class(self):
    with StringIO() as fh:
        libcloud.enable_debug(fh)
        conn = Connection(timeout=10)
        conn.connect()
    self.assertTrue(isinstance(conn.connection, LoggingConnection))