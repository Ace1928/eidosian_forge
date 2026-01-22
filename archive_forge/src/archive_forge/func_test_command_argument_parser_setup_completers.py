from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import time
import six
from six.moves import input
import boto
import sys
import gslib
from gslib import command_runner
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.command_runner import CommandRunner
from gslib.command_runner import HandleArgCoding
from gslib.command_runner import HandleHeaderCoding
from gslib.exception import CommandException
from gslib.tab_complete import CloudObjectCompleter
from gslib.tab_complete import CloudOrLocalObjectCompleter
from gslib.tab_complete import LocalObjectCompleter
from gslib.tab_complete import LocalObjectOrCannedACLCompleter
from gslib.tab_complete import NoOpCompleter
import gslib.tests.testcase as testcase
import gslib.tests.util as util
from gslib.tests.util import ARGCOMPLETE_AVAILABLE
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.utils import system_util
from gslib.utils.constants import GSUTIL_PUB_TARBALL
from gslib.utils.text_util import InsistAscii
from gslib.utils.unit_util import SECONDS_PER_DAY
from six import add_move, MovedModule
from six.moves import mock
@unittest.skipUnless(ARGCOMPLETE_AVAILABLE, 'Tab completion requires argcomplete')
def test_command_argument_parser_setup_completers(self):
    command_map = {FakeCommandWithCompleters.command_spec.command_name: FakeCommandWithCompleters()}
    runner = CommandRunner(bucket_storage_uri_class=self.mock_bucket_storage_uri, gsutil_api_class_map_factory=self.mock_gsutil_api_class_map_factory, command_map=command_map)
    main_parser = FakeArgparseParser()
    runner.ConfigureCommandArgumentParsers(main_parser)
    self.assertEqual(1, len(main_parser.subparsers.parsers))
    subparser = main_parser.subparsers.parsers[0]
    self.assertEqual(6, len(subparser.arguments))
    self.assertEqual(CloudObjectCompleter, type(subparser.arguments[0].completer))
    self.assertEqual(LocalObjectCompleter, type(subparser.arguments[1].completer))
    self.assertEqual(CloudOrLocalObjectCompleter, type(subparser.arguments[2].completer))
    self.assertEqual(NoOpCompleter, type(subparser.arguments[3].completer))
    self.assertEqual(CloudObjectCompleter, type(subparser.arguments[4].completer))
    self.assertEqual(LocalObjectOrCannedACLCompleter, type(subparser.arguments[5].completer))