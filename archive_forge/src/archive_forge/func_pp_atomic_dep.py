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
def pp_atomic_dep(dep):
    s = dep['name']
    if dep.get('archqual') is not None:
        s += ':%s' % dep['archqual']
    v = dep.get('version')
    if v is not None:
        s += ' (%s %s)' % v
    a = dep.get('arch')
    if a is not None:
        s += ' [%s]' % ' '.join(map(pp_arch, a))
    r = dep.get('restrictions')
    if r is not None:
        s += ' %s' % ' '.join(map(pp_restrictions, r))
    return s