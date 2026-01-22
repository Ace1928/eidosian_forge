from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_file_completion(self):
    with TempDir(prefix='test_dir_fc', dir='.'):
        fc = FilesCompleter()
        os.makedirs(os.path.join('abcdefж', 'klm'))
        self.assertEqual(fc('a'), ['abcdefж/'])
        os.makedirs(os.path.join('abcaha', 'klm'))
        with open('abcxyz', 'w') as fp:
            fp.write('test')
        self.assertEqual(set(fc('a')), set(['abcdefж/', 'abcaha/', 'abcxyz']))