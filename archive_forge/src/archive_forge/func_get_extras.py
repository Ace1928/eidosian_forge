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
def get_extras(requested, available):
    result = set()
    requested = set(requested or [])
    available = set(available or [])
    if '*' in requested:
        requested.remove('*')
        result |= available
    for r in requested:
        if r == '-':
            result.add(r)
        elif r.startswith('-'):
            unwanted = r[1:]
            if unwanted not in available:
                logger.warning('undeclared extra: %s' % unwanted)
            if unwanted in result:
                result.remove(unwanted)
        else:
            if r not in available:
                logger.warning('undeclared extra: %s' % r)
            result.add(r)
    return result