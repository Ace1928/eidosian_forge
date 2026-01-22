import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def test_send_with_disconnected_socket(self):

    class DisconnectedSocket:

        def __init__(self, err):
            self.err = err

        def send(self, content):
            raise self.err

        def close(self):
            pass
    errs = []
    for err_cls in (IOError, socket.error):
        for errnum in osutils._end_of_stream_errors:
            errs.append(err_cls(errnum))
    for err in errs:
        sock = DisconnectedSocket(err)
        self.assertRaises(errors.ConnectionReset, osutils.send_all, sock, b'some more content')