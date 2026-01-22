import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
def create_tempfiles(self, files, ext='.conf'):
    tempfiles = []
    for basename, contents in files:
        if not os.path.isabs(basename):
            tmpdir = tempfile.mkdtemp()
            path = os.path.join(tmpdir, basename + ext)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
        else:
            path = basename + ext
        fd = os.open(path, os.O_CREAT | os.O_WRONLY)
        tempfiles.append(path)
        try:
            os.write(fd, contents.encode('utf-8'))
        finally:
            os.close(fd)
    return tempfiles