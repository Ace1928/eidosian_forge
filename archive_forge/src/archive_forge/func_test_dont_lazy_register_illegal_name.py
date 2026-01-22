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
def test_dont_lazy_register_illegal_name(self):
    self.assertRaises(config.IllegalOptionName, self.registry.register_lazy, ' foo', self.__module__, 'TestOptionRegistry.lazy_option')
    self.assertRaises(config.IllegalOptionName, self.registry.register_lazy, '1,2', self.__module__, 'TestOptionRegistry.lazy_option')