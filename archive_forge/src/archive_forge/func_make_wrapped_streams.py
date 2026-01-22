import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
def make_wrapped_streams(self, stdout_encoding, stderr_encoding, stdin_encoding, user_encoding='user_encoding', enable_fake_encodings=True):
    sys.stdout = StringIOWithEncoding()
    sys.stdout.encoding = stdout_encoding
    sys.stderr = StringIOWithEncoding()
    sys.stderr.encoding = stderr_encoding
    sys.stdin = StringIOWithEncoding()
    sys.stdin.encoding = stdin_encoding
    osutils._cached_user_encoding = user_encoding
    if enable_fake_encodings:
        fake_codec.add(stdout_encoding)
        fake_codec.add(stderr_encoding)
        fake_codec.add(stdin_encoding)