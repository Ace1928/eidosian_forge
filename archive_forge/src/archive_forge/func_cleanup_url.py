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
def cleanup_url(url):
    tmp = url.rstrip('/')
    filename = tmp.split('/')[-1]
    if url.endswith('/'):
        display_url = tmp + '/'
    else:
        display_url = url
    return (filename, display_url)