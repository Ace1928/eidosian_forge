import json
import os
from unittest import mock
import testtools
from openstack.baremetal import configdrive
def test_with_vendor_data(self):
    self._check({'foo': 42}, vendor_data={'foo': 'bar'})