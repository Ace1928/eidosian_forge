import argparse
import copy
import os
from unittest import mock
import fixtures
import testtools
import yaml
from openstack import config
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
def test_fallthrough(self):
    c = config.OpenStackConfig(config_files=[self.no_yaml], vendor_files=[self.no_yaml], secure_files=[self.no_yaml])
    for k in os.environ.keys():
        if k.startswith('OS_'):
            self.useFixture(fixtures.EnvironmentVariable(k))
    c.get_one(cloud='defaults', validate=False)