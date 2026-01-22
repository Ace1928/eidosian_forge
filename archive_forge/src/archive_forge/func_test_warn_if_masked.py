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
def test_warn_if_masked(self):
    warnings = []

    def warning(*args):
        warnings.append(args[0] % args[1:])
    self.overrideAttr(trace, 'warning', warning)

    def set_option(store, warn_masked=True):
        warnings[:] = []
        conf.set_user_option('example_option', repr(store), store=store, warn_masked=warn_masked)

    def assertWarning(warning):
        if warning is None:
            self.assertEqual(0, len(warnings))
        else:
            self.assertEqual(1, len(warnings))
            self.assertEqual(warning, warnings[0])
    branch = self.make_branch('.')
    conf = branch.get_config()
    set_option(config.STORE_GLOBAL)
    assertWarning(None)
    set_option(config.STORE_BRANCH)
    assertWarning(None)
    set_option(config.STORE_GLOBAL)
    assertWarning('Value "4" is masked by "3" from branch.conf')
    set_option(config.STORE_GLOBAL, warn_masked=False)
    assertWarning(None)
    set_option(config.STORE_LOCATION)
    assertWarning(None)
    set_option(config.STORE_BRANCH)
    assertWarning('Value "3" is masked by "0" from locations.conf')
    set_option(config.STORE_BRANCH, warn_masked=False)
    assertWarning(None)