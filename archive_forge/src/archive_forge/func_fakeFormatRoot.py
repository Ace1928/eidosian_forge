from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def fakeFormatRoot(self, obj: object) -> str:
    return 'R(%s)' % obj