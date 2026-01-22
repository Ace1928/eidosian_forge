import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def checkValue(self, key, value, type='string', list=False):
    if list == True:
        values = self.getList(value)
    else:
        values = [value]
    for value in values:
        if type == 'string':
            code = self.checkString(value)
        if type == 'localestring':
            continue
        elif type == 'boolean':
            code = self.checkBoolean(value)
        elif type == 'numeric':
            code = self.checkNumber(value)
        elif type == 'integer':
            code = self.checkInteger(value)
        elif type == 'regex':
            code = self.checkRegex(value)
        elif type == 'point':
            code = self.checkPoint(value)
        if code == 1:
            self.errors.append("'%s' is not a valid %s" % (value, type))
        elif code == 2:
            self.warnings.append("Value of key '%s' is deprecated" % key)