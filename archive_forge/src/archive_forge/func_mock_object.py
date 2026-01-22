import os
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from glance_store._drivers import rbd as rbd_store
from glance_store._drivers import swift
from glance_store import location
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_db import options
from oslo_serialization import jsonutils
from glance.tests import stubs
from glance.tests import utils as test_utils
def mock_object(self, obj, attr_name, *args, **kwargs):
    """Use python mock to mock an object attribute

        Mocks the specified objects attribute with the given value.
        Automatically performs 'addCleanup' for the mock.
        """
    patcher = mock.patch.object(obj, attr_name, *args, **kwargs)
    result = patcher.start()
    self.addCleanup(patcher.stop)
    return result