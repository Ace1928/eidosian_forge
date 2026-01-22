from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_completers(self):

    def c_url(prefix, parsed_args, **kwargs):
        return ['http://url1', 'http://url2']

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('--url').completer = c_url
        parser.add_argument('--email', nargs=3, choices=['a@b.c', 'a@b.d', 'ab@c.d', 'bcd@e.f', 'bce@f.g'])
        return parser
    expected_outputs = (('prog --url ', ['http://url1', 'http://url2']), ('prog --url "', ['http://url1', 'http://url2']), ('prog --url "http://url1" --email ', ['a@b.c', 'a@b.d', 'ab@c.d', 'bcd@e.f', 'bce@f.g']), ('prog --url "http://url1" --email a', ['a@b.c', 'a@b.d', 'ab@c.d']), ('prog --url "http://url1" --email "a@', ['a@b.c', 'a@b.d']), ('prog --url "http://url1" --email "a@b.c" "a@b.d" "a@', ['a@b.c', 'a@b.d']), ('prog --url "http://url1" --email "a@b.c" "a@b.d" "ab@c.d" ', ['--url', '--email', '-h', '--help']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))