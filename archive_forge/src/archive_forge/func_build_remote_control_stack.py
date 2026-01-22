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
def build_remote_control_stack(test):
    transport_class, server_class = transport_remote.get_test_permutations()[0]
    build_backing_branch(test, 'branch', transport_class, server_class)
    b = branch.Branch.open(test.get_url('branch'))
    return config.RemoteControlStack(b.controldir)