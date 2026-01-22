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
def test_load_erroneous_content(self):
    """Ensure we display a proper error on content that can't be parsed."""
    with open('foo.conf', 'wb') as f:
        f.write(b'[open_section\n')
    conf = config.IniBasedConfig(file_name='foo.conf')
    self.assertRaises(config.ParseConfigError, conf._get_parser)