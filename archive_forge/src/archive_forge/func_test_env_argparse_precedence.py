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
def test_env_argparse_precedence(self):
    self.useFixture(fixtures.EnvironmentVariable('OS_TENANT_NAME', 'tenants-are-bad'))
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one(cloud='envvars', argparse=self.options, validate=False)
    self.assertEqual(cc.auth['project_name'], 'project')