from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_optional_nargs(self):

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('--foo', choices=['foo1', 'foo2'], nargs=2)
        parser.add_argument('--bar', choices=['bar1', 'bar2'], nargs='?')
        parser.add_argument('--baz', choices=['baz1', 'baz2'], nargs='*')
        parser.add_argument('--qux', choices=['qux1', 'qux2'], nargs='+')
        parser.add_argument('--foobar', choices=['pos', '--opt'], nargs=argparse.REMAINDER)
        return parser
    options = ['--foo', '--bar', '--baz', '--qux', '--foobar', '-h', '--help']
    expected_outputs = (('prog ', options), ('prog --foo ', ['foo1', 'foo2']), ('prog --foo foo1 ', ['foo1', 'foo2']), ('prog --foo foo1 foo2 ', options), ('prog --bar ', ['bar1', 'bar2'] + options), ('prog --bar bar1 ', options), ('prog --baz ', ['baz1', 'baz2'] + options), ('prog --baz baz1 ', ['baz1', 'baz2'] + options), ('prog --qux ', ['qux1', 'qux2']), ('prog --qux qux1 ', ['qux1', 'qux2'] + options), ('prog --foobar ', ['pos', '--opt']), ('prog --foobar pos ', ['pos', '--opt']), ('prog --foobar --', ['--opt ']), ('prog --foobar --opt ', ['pos', '--opt']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))