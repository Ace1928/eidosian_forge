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
def test_locations_matching_sections(self):
    loc_config = self.locations_config
    loc_config.set_user_option('file', 'locations')
    parser = loc_config._get_parser()
    location_names = self.tree.basedir.split('/')
    parent = '/'.join(location_names[:-1])
    child = '/'.join(location_names + ['child'])
    parser[parent] = {}
    parser[parent]['file'] = 'parent'
    parser[child] = {}
    parser[child]['file'] = 'child'
    self.assertSectionNames([self.tree.basedir, parent], loc_config)