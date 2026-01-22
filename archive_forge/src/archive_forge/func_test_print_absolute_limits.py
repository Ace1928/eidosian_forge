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
def test_print_absolute_limits(self):
    limits = [TestAbsoluteLimits('maxTotalPrivateNetworks', 3), TestAbsoluteLimits('totalPrivateNetworksUsed', 0), TestAbsoluteLimits('maxImageMeta', 15), TestAbsoluteLimits('totalCoresUsed', 10), TestAbsoluteLimits('totalInstancesUsed', 5), TestAbsoluteLimits('maxServerMeta', 10), TestAbsoluteLimits('totalRAMUsed', 10240), TestAbsoluteLimits('totalFloatingIpsUsed', 10)]
    novaclient.v2.shell._print_absolute_limits(limits=limits)