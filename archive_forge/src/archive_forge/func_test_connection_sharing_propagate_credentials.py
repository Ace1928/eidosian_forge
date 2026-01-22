import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
def test_connection_sharing_propagate_credentials(self):
    t = transport.ConnectedTransport('ftp://user@host.com/abs/path')
    self.assertEqual('user', t._parsed_url.user)
    self.assertEqual('host.com', t._parsed_url.host)
    self.assertIs(None, t._get_connection())
    self.assertIs(None, t._parsed_url.password)
    c = t.clone('subdir')
    self.assertIs(None, c._get_connection())
    self.assertIs(None, t._parsed_url.password)
    password = 'secret'
    connection = object()
    t._set_connection(connection, password)
    self.assertIs(connection, t._get_connection())
    self.assertIs(password, t._get_credentials())
    self.assertIs(connection, c._get_connection())
    self.assertIs(password, c._get_credentials())
    new_password = 'even more secret'
    c._update_credentials(new_password)
    self.assertIs(connection, t._get_connection())
    self.assertIs(new_password, t._get_credentials())
    self.assertIs(connection, c._get_connection())
    self.assertIs(new_password, c._get_credentials())