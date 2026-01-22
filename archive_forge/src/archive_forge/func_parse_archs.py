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
def parse_archs(raw):
    archs = []
    for arch in cls.__blank_sep_RE.split(raw.strip()):
        disabled = arch[0] == '!'
        if disabled:
            arch = arch[1:]
        archs.append(cls.ArchRestriction(not disabled, arch))
    return archs