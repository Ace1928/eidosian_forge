import argparse
import mimetypes
import os
import posixpath
import shutil
import stat
import tempfile
from importlib.util import find_spec
from urllib.request import build_opener
import django
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.core.management.utils import (
from django.template import Context, Engine
from django.utils import archive
from django.utils.http import parse_header_parameters
from django.utils.version import get_docs_version
def make_writeable(self, filename):
    """
        Make sure that the file is writeable.
        Useful if our source is read-only.
        """
    if not os.access(filename, os.W_OK):
        st = os.stat(filename)
        new_permissions = stat.S_IMODE(st.st_mode) | stat.S_IWUSR
        os.chmod(filename, new_permissions)