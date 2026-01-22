import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test__pretty_print(self):
    func = mock.Mock(return_value=None)
    self.cmd_base.verbose = True
    self.assertIsNone(self.cmd_base._pretty_print(func))
    self.cmd_base.verbose = False
    self.assertIsNone(self.cmd_base._pretty_print(func))