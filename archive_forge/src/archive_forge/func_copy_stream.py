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
def copy_stream(self, instream, outfile, encoding=None):
    assert not os.path.isdir(outfile)
    self.ensure_dir(os.path.dirname(outfile))
    logger.info('Copying stream %s to %s', instream, outfile)
    if not self.dry_run:
        if encoding is None:
            outstream = open(outfile, 'wb')
        else:
            outstream = codecs.open(outfile, 'w', encoding=encoding)
        try:
            shutil.copyfileobj(instream, outstream)
        finally:
            outstream.close()
    self.record_as_written(outfile)