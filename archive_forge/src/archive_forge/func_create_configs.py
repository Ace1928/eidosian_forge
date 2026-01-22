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
def create_configs(test):
    """Create configuration files for a given test.

    This requires creating a tree (and populate the ``test.tree`` attribute)
    and its associated branch and will populate the following attributes:

    - branch_config: A BranchConfig for the associated branch.

    - locations_config : A LocationConfig for the associated branch

    - breezy_config: A GlobalConfig.

    The tree and branch are created in a 'tree' subdirectory so the tests can
    still use the test directory to stay outside of the branch.
    """
    tree = test.make_branch_and_tree('tree')
    test.tree = tree
    test.branch_config = config.BranchConfig(tree.branch)
    test.locations_config = config.LocationConfig(tree.basedir)
    test.breezy_config = config.GlobalConfig()