from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_readline_entry_point(self):

    def get_readline_completions(completer, text):
        completions = []
        for i in range(9999):
            completion = completer.rl_complete(text, i)
            if completion is None:
                break
            completions.append(completion)
        return completions
    parser = ArgumentParser()
    parser.add_argument('rover', choices=['sojourner', 'spirit', 'opportunity', 'curiosity'])
    parser.add_argument('antenna', choices=['low gain', 'high gain'])
    completer = CompletionFinder(parser)
    self.assertEqual(get_readline_completions(completer, ''), ['-h', '--help', 'sojourner', 'spirit', 'opportunity', 'curiosity'])
    self.assertEqual(get_readline_completions(completer, 's'), ['sojourner', 'spirit'])
    self.assertEqual(get_readline_completions(completer, 'x'), [])