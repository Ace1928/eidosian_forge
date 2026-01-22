import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def get_terminal_encoding(trace=False):
    """Find the best encoding for printing to the screen.

    This attempts to check both sys.stdout and sys.stdin to see
    what encoding they are in, and if that fails it falls back to
    osutils.get_user_encoding().
    The problem is that on Windows, locale.getpreferredencoding()
    is not the same encoding as that used by the console:
    http://mail.python.org/pipermail/python-list/2003-May/162357.html

    On my standard US Windows XP, the preferred encoding is
    cp1252, but the console is cp437

    :param trace: If True trace the selected encoding via mutter().
    """
    from .trace import mutter
    output_encoding = getattr(sys.stdout, 'encoding', None)
    if not output_encoding:
        input_encoding = getattr(sys.stdin, 'encoding', None)
        if not input_encoding:
            output_encoding = get_user_encoding()
            if trace:
                mutter('encoding stdout as osutils.get_user_encoding() %r', output_encoding)
        else:
            output_encoding = input_encoding
            if trace:
                mutter('encoding stdout as sys.stdin encoding %r', output_encoding)
    elif trace:
        mutter('encoding stdout as sys.stdout encoding %r', output_encoding)
    if output_encoding == 'cp0':
        output_encoding = get_user_encoding()
        if trace:
            mutter('cp0 is invalid encoding. encoding stdout as osutils.get_user_encoding() %r', output_encoding)
    try:
        codecs.lookup(output_encoding)
    except LookupError:
        sys.stderr.write('brz: warning: unknown terminal encoding %s.\n  Using encoding %s instead.\n' % (output_encoding, get_user_encoding()))
        output_encoding = get_user_encoding()
    return output_encoding