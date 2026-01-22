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
def test_register_argparse_not_password(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    parser = argparse.ArgumentParser()
    args = ['--os-auth-type', 'v3token', '--os-token', 'some-secret']
    c.register_argparse_arguments(parser, args)
    opts, _remain = parser.parse_known_args(args)
    self.assertEqual(opts.os_token, 'some-secret')