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
def test_set_unset_default_stack_on(self):
    my_dir = self.make_controldir('.')
    bzrdir_config = config.BzrDirConfig(my_dir)
    self.assertIs(None, bzrdir_config.get_default_stack_on())
    bzrdir_config.set_default_stack_on('Foo')
    self.assertEqual('Foo', bzrdir_config._config.get_option('default_stack_on'))
    self.assertEqual('Foo', bzrdir_config.get_default_stack_on())
    bzrdir_config.set_default_stack_on(None)
    self.assertIs(None, bzrdir_config.get_default_stack_on())