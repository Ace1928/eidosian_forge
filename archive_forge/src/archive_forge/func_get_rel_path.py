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
def get_rel_path(root, path):
    root = root.replace(os.path.sep, '/')
    path = path.replace(os.path.sep, '/')
    assert path.startswith(root)
    return path[len(root):].lstrip('/')