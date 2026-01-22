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
def test_remove_new_option(self):
    a_dict = dict()
    section = self.get_section(a_dict)
    section.set('foo', 'bar')
    section.remove('foo')
    self.assertFalse('foo' in section.options)
    self.assertTrue('foo' in section.orig)
    self.assertEqual(config._NewlyCreatedOption, section.orig['foo'])