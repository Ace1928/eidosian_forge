import builtins
import errno
import hashlib
import io
import json
import os
import stat
from unittest import mock
import uuid
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import filesystem
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_configure_add_same_dir_multiple_times(self):
    """
        Tests BadStoreConfiguration exception is raised if same directory
        is specified multiple times in filesystem_store_datadirs.
        """
    store_map = [self.useFixture(fixtures.TempDir()).path, self.useFixture(fixtures.TempDir()).path]
    self.conf.clear_override('filesystem_store_datadir', group='glance_store')
    self.conf.set_override('filesystem_store_datadirs', [store_map[0] + ':100', store_map[1] + ':200', store_map[0] + ':300'], group='glance_store')
    self.assertRaises(exceptions.BadStoreConfiguration, self.store.configure_add)