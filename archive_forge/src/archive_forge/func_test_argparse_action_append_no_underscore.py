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
def test_argparse_action_append_no_underscore(self):
    c = config.OpenStackConfig(config_files=[self.no_yaml], vendor_files=[self.no_yaml], secure_files=[self.no_yaml])
    parser = argparse.ArgumentParser()
    parser.add_argument('--foo', action='append')
    argv = ['--foo', '1', '--foo', '2']
    c.register_argparse_arguments(parser, argv=argv)
    opts, _remain = parser.parse_known_args(argv)
    self.assertEqual(opts.foo, ['1', '2'])