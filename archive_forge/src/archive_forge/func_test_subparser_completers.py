from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_subparser_completers(self):

    def c_depends_on_positional_arg1(prefix, parsed_args, **kwargs):
        return [parsed_args.arg1]

    def c_depends_on_optional_arg5(prefix, parsed_args, **kwargs):
        return [parsed_args.arg5]

    def make_parser():
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        subparser = subparsers.add_parser('subcommand')
        subparser.add_argument('arg1')
        subparser.add_argument('arg2').completer = c_depends_on_positional_arg1
        subparser.add_argument('arg3').completer = c_depends_on_optional_arg5
        subparser.add_argument('--arg4').completer = c_depends_on_optional_arg5
        subparser.add_argument('--arg5')
        return parser
    expected_outputs = (('prog subcommand val1 ', ['val1', '--arg4', '--arg5', '-h', '--help']), ('prog subcommand val1 val2 --arg5 val5 ', ['val5', '--arg4', '--arg5', '-h', '--help']), ('prog subcommand val1 val2 --arg5 val6 --arg4 v', ['val6 ']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))