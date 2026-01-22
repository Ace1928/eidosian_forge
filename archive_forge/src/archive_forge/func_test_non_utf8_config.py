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
def test_non_utf8_config(self):
    conf = config.AuthenticationConfig(_file=BytesIO(b'foo = bar\xff'))
    self.assertRaises(config.ConfigContentError, conf._get_config)