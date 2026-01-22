from argparse import ArgumentParser
from argparse import ArgumentTypeError
from unittest import mock
from datetime import datetime
from testtools import ExpectedException
from vitrageclient.tests.cli.base import CliTestCase
from vitrageclient.v1.cli.event import EventPost
def test_iso8601_parsing_with_wrong_date_format(self):
    self.assertRaises(ArgumentTypeError, self.event_post.iso8601, '2014/12/13 12:44:21')