import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def hasKey(self, key, group=None):
    if not group:
        group = self.defaultGroup
    return key in self.content[group]