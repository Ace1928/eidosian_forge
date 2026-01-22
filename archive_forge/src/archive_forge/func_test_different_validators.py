from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_different_validators(self):

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('var', choices=['bus', 'car'])
        parser.add_argument('value', choices=['orange', 'apple'])
        return parser
    validators = (lambda x, y: False, lambda x, y: True, lambda x, y: x.startswith(y))
    expected_outputs = (('prog ', ['-h', '--help'], validators[0]), ('prog ', ['bus', 'car', '-h', '--help'], validators[1]), ('prog bu', ['bus', 'car'], validators[1]), ('prog bus ', ['apple', 'orange', '-h', '--help'], validators[1]), ('prog bus appl', ['apple '], validators[2]), ('prog bus cappl', [''], validators[2]), ('prog bus pple ', ['-h', '--help'], validators[2]))
    for cmd, output, validator in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd, validator=validator)), set(output))