from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_subparsers(self):

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('--age', type=int)
        sub = parser.add_subparsers()
        eggs = sub.add_parser('eggs')
        eggs.add_argument('type', choices=['on a boat', 'with a goat', 'in the rain', 'on a train'])
        spam = sub.add_parser('spam')
        spam.add_argument('type', choices=['ham', 'iberico'])
        return parser
    expected_outputs = (('prog ', ['--help', 'eggs', '-h', 'spam', '--age']), ('prog --age 1 eggs', ['eggs ']), ('prog --age 2 eggs ', ['on\\ a\\ train', 'with\\ a\\ goat', 'on\\ a\\ boat', 'in\\ the\\ rain', '--help', '-h']), ('prog eggs ', ['on\\ a\\ train', 'with\\ a\\ goat', 'on\\ a\\ boat', 'in\\ the\\ rain', '--help', '-h']), ('prog eggs "on a', ['on a train', 'on a boat']), ('prog eggs on\\ a', ['on\\ a\\ train', 'on\\ a\\ boat']), ('prog spam ', ['iberico', 'ham', '--help', '-h']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))
        self.assertEqual(set(self.run_completer(make_parser(), cmd, exclude=['-h'])), set(output) - set(['-h']))
        self.assertEqual(set(self.run_completer(make_parser(), cmd, exclude=['-h', '--help'])), set(output) - set(['-h', '--help']))