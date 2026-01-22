import argparse
import base64
import builtins
import collections
import datetime
import io
import os
from unittest import mock
import fixtures
from oslo_utils import timeutils
import testtools
import novaclient
from novaclient import api_versions
from novaclient import base
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
import novaclient.v2.shell
def test_rebuild_with_trusted_image_certificates_envar(self):
    self.useFixture(fixtures.EnvironmentVariable('OS_TRUSTED_IMAGE_CERTIFICATE_IDS', 'var_id1,var_id2'))
    self.run_command('rebuild sample-server %s' % FAKE_UUID_1, api_version='2.63')
    self.assert_called('GET', '/servers?name=sample-server', pos=0)
    self.assert_called('GET', '/servers/1234', pos=1)
    self.assert_called('GET', '/v2/images/%s' % FAKE_UUID_1, pos=2)
    self.assert_called('POST', '/servers/1234/action', {'rebuild': {'imageRef': FAKE_UUID_1, 'description': None, 'trusted_image_certificates': ['var_id1', 'var_id2']}}, pos=3)
    self.assert_called('GET', '/v2/images/%s' % FAKE_UUID_2, pos=4)