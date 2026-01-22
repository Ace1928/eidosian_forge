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
def get_resources_dests(resources_root, rules):
    """Find destinations for resources files"""

    def get_rel_path(root, path):
        root = root.replace(os.path.sep, '/')
        path = path.replace(os.path.sep, '/')
        assert path.startswith(root)
        return path[len(root):].lstrip('/')
    destinations = {}
    for base, suffix, dest in rules:
        prefix = os.path.join(resources_root, base)
        for abs_base in iglob(prefix):
            abs_glob = os.path.join(abs_base, suffix)
            for abs_path in iglob(abs_glob):
                resource_file = get_rel_path(resources_root, abs_path)
                if dest is None:
                    destinations.pop(resource_file, None)
                else:
                    rel_path = get_rel_path(abs_base, abs_path)
                    rel_dest = dest.replace(os.path.sep, '/').rstrip('/')
                    destinations[resource_file] = rel_dest + '/' + rel_path
    return destinations