import argparse
from unittest import mock
import testtools
from ironicclient.osc import plugin
from ironicclient.tests.unit.osc import fakes
from ironicclient.v1 import client
@mock.patch.object(plugin, 'OS_BAREMETAL_API_LATEST', new=False)
def test___call___latest(self):
    parser = argparse.ArgumentParser()
    plugin.build_option_parser(parser)
    namespace = argparse.Namespace()
    parser.parse_known_args(['--os-baremetal-api-version', 'latest'], namespace)
    self.assertEqual(plugin.LATEST_VERSION, namespace.os_baremetal_api_version)
    self.assertTrue(plugin.OS_BAREMETAL_API_LATEST)