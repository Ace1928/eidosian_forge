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
def inc_convert(self, value):
    """Default converter for the inc:// protocol."""
    if not os.path.isabs(value):
        value = os.path.join(self.base, value)
    with codecs.open(value, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result