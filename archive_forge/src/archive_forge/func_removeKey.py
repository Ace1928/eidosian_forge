import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def removeKey(self, key, group=None, locales=True):
    if not group:
        group = self.defaultGroup
    try:
        if locales:
            for name in list(self.content[group]):
                if re.match('^' + key + xdg.Locale.regex + '$', name) and name != key:
                    del self.content[group][name]
        value = self.content[group].pop(key)
        self.tainted = True
        return value
    except KeyError as e:
        if debug:
            if e == group:
                raise NoGroupError(group, self.filename)
            else:
                raise NoKeyError(key, group, self.filename)
        else:
            return ''