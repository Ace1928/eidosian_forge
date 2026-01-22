import codecs
from collections import deque
import contextlib
import csv
from glob import iglob as std_iglob
import io
import json
import logging
import os
import py_compile
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from . import DistlibException
from .compat import (string_types, text_type, shutil, raw_input, StringIO,
def parse_name_and_version(p):
    """
    A utility method used to get name and version from a string.

    From e.g. a Provides-Dist value.

    :param p: A value in a form 'foo (1.0)'
    :return: The name and version as a tuple.
    """
    m = NAME_VERSION_RE.match(p)
    if not m:
        raise DistlibException("Ill-formed name/version string: '%s'" % p)
    d = m.groupdict()
    return (d['name'].strip().lower(), d['ver'])