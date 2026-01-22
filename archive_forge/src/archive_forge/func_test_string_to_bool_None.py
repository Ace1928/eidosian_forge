import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
def test_string_to_bool_None(self):
    self.assertIsNone(utils.str2bool(None))