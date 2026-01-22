import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test__require(self):
    self.assertRaises(common.ArgumentRequired, self.cmd_base._require, 'attr_1')
    self.cmd_base.attr_1 = None
    self.assertRaises(common.ArgumentRequired, self.cmd_base._require, 'attr_1')
    self.cmd_base.attr_1 = 'attr_v1'
    self.cmd_base._require('attr_1')