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
def test_not_supported_callable_default_value_not_unicode(self):

    def bar_not_unicode():
        return b'bar'
    opt = config.Option('foo', default=bar_not_unicode)
    self.assertRaises(AssertionError, opt.get_default)