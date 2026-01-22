import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test__parse_options(self):
    parser = optparse.OptionParser()
    parser.add_option('--%s' % 'test_1', default='test_1v')
    parser.add_option('--%s' % 'test_2', default='test_2v')
    self.cmd_base._parse_options(parser)
    self.assertEqual('test_1v', self.cmd_base.test_1)
    self.assertEqual('test_2v', self.cmd_base.test_2)