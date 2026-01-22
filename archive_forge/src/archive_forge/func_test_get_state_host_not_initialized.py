import sys
from unittest import mock
import fixtures
from oslo_concurrency import processutils
from oslo_config import cfg
from oslotest import base
from glance_store import exceptions
def test_get_state_host_not_initialized(self):
    self.__manager__.state = None
    self.assertRaises(exceptions.HostNotInitialized, self.get_state)