import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test__make_list(self):
    self.assertRaises(AttributeError, self.cmd_base._make_list, 'attr1')
    self.cmd_base.attr1 = 'v1,v2'
    self.cmd_base._make_list('attr1')
    self.assertEqual(['v1', 'v2'], self.cmd_base.attr1)
    self.cmd_base.attr1 = ['v3']
    self.cmd_base._make_list('attr1')
    self.assertEqual(['v3'], self.cmd_base.attr1)