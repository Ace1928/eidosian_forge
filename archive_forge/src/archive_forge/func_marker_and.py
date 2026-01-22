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
def marker_and(remaining):
    lhs, remaining = marker_expr(remaining)
    while remaining:
        m = AND.match(remaining)
        if not m:
            break
        remaining = remaining[m.end():]
        rhs, remaining = marker_expr(remaining)
        lhs = {'op': 'and', 'lhs': lhs, 'rhs': rhs}
    return (lhs, remaining)