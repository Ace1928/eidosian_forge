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
def test_add_file_too_big(self):
    """
        Tests that adding an excessively large image file
        raises an appropriate exception
        """
    self._do_test_add_write_failure(errno.EFBIG, exceptions.StorageFull)