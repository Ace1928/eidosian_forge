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
def test_get_aliases(self):
    my_config = self._get_sample_config()
    aliases = my_config.get_aliases()
    self.assertEqual(2, len(aliases))
    sorted_keys = sorted(aliases)
    self.assertEqual('help', aliases[sorted_keys[0]])
    self.assertEqual(sample_long_alias, aliases[sorted_keys[1]])