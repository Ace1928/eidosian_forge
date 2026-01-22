import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
def test_parse_username(self):
    self.assertEqual(('', 'jdoe@example.com'), config.parse_username('jdoe@example.com'))
    self.assertEqual(('', 'jdoe@example.com'), config.parse_username('<jdoe@example.com>'))
    self.assertEqual(('John Doe', 'jdoe@example.com'), config.parse_username('John Doe <jdoe@example.com>'))
    self.assertEqual(('John Doe', ''), config.parse_username('John Doe'))
    self.assertEqual(('John Doe', 'jdoe@example.com'), config.parse_username('John Doe jdoe@example.com'))
    self.assertEqual(('John Doe', 'jdoe[bot]@example.com'), config.parse_username('John Doe <jdoe[bot]@example.com>'))