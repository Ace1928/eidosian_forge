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
def test_duplicate_section_name_error_line(self):
    try:
        co = configobj.ConfigObj(BytesIO(erroneous_config), raise_errors=True)
    except config.configobj.DuplicateError as e:
        self.assertEqual(3, e.line_number)
    else:
        self.fail('Error in config file not detected')