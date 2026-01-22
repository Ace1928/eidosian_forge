from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_repl_subparser_parse_after_complete(self):
    p = ArgumentParser()
    sp = p.add_subparsers().add_parser('foo')
    sp.add_argument('bar', choices=['bar'])
    c = CompletionFinder(p, always_complete_options=True)
    completions = self.run_completer(p, c, 'prog foo ')
    assert set(completions) == set(['-h', '--help', 'bar'])
    args = p.parse_args(['foo', 'bar'])
    assert args.bar == 'bar'
    with self.assertRaises(SystemExit):
        p.parse_args(['foo'])