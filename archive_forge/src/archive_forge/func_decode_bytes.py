import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
def decode_bytes(self, value):
    try:
        return value.decode(self.encoding)
    except UnicodeDecodeError as e:
        logger.warning('decoding from %s failed; attempting to detect the true encoding', self.encoding)
        result = chardet.detect(value)
        try:
            decoded = value.decode(result['encoding'])
            self.encoding = result['encoding']
            return decoded
        except UnicodeDecodeError:
            raise e