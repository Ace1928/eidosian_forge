from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_nargs_one_or_more(self):

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('h1', choices=['c', 'd'])
        parser.add_argument('var', choices=['bus', 'car'], nargs='+')
        parser.add_argument('value', choices=['orange', 'apple'])
        parser.add_argument('end', choices=['end'])
        return parser
    expected_outputs = (('prog ', ['c', 'd', '-h', '--help']), ('prog c ', ['bus', 'car', '-h', '--help']), ('prog c bu', ['bus ']), ('prog c bus ', ['bus', 'car', 'apple', 'orange', '-h', '--help']), ('prog c bus car ', ['bus', 'car', 'apple', 'orange', '-h', '--help']), ('prog c bus appl', ['apple ']), ('prog c bus apple ', ['bus', 'car', 'apple', 'orange', 'end', '-h', '--help']), ('prog c bus car apple ', ['bus', 'car', 'apple', 'orange', 'end', '-h', '--help']), ('prog c bus car apple end ', ['bus', 'car', 'apple', 'orange', 'end', '-h', '--help']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))