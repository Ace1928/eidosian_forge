from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_repl_reuse_parser_with_positional(self):
    p = ArgumentParser()
    p.add_argument('foo', choices=['aa', 'bb', 'cc'])
    p.add_argument('bar', choices=['d', 'e'])
    c = CompletionFinder(p, always_complete_options=True)
    self.assertEqual(set(self.run_completer(p, c, 'prog ')), set(['-h', '--help', 'aa', 'bb', 'cc']))
    self.assertEqual(set(self.run_completer(p, c, 'prog aa ')), set(['-h', '--help', 'd', 'e']))
    self.assertEqual(set(self.run_completer(p, c, 'prog ')), set(['-h', '--help', 'aa', 'bb', 'cc']))