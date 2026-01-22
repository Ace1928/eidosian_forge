import argparse
from unittest import mock
import testtools
from aodhclient.v2 import alarm_cli
def test_validate_time_constraint_with_bad_format(self):
    string = 'name=const2;start="0 11 * * *";duration:2'
    self.assertRaises(argparse.ArgumentTypeError, self.cli_alarm_create.validate_time_constraint, string)