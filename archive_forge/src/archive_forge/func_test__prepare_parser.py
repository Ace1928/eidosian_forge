import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test__prepare_parser(self):
    parser = optparse.OptionParser()
    common.CommandsBase.params = ['test_1', 'test_2']
    self.cmd_base._prepare_parser(parser)
    option = parser.get_option('--%s' % 'test_1')
    self.assertIsNotNone(option)
    option = parser.get_option('--%s' % 'test_2')
    self.assertIsNotNone(option)