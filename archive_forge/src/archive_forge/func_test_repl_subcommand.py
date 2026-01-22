from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_repl_subcommand(self):
    p = ArgumentParser()
    p.add_argument('--foo')
    p.add_argument('--bar')
    s = p.add_subparsers()
    s.add_parser('list')
    s.add_parser('set')
    show = s.add_parser('show')

    def abc():
        pass
    show.add_argument('--test')
    ss = show.add_subparsers()
    de = ss.add_parser('depth')
    de.set_defaults(func=abc)
    c = CompletionFinder(p, always_complete_options=True)
    expected_outputs = (('prog ', ['-h', '--help', '--foo', '--bar', 'list', 'show', 'set']), ('prog li', ['list ']), ('prog s', ['show', 'set']), ('prog show ', ['--test', 'depth', '-h', '--help']), ('prog show d', ['depth ']), ('prog show depth ', ['-h', '--help']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(p, c, cmd)), set(output))