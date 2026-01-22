from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def make_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-1', choices=['bar<$>baz'])
    parser.add_argument('-2', choices=['\\* '])
    parser.add_argument('-3', choices=['"\''])
    return parser