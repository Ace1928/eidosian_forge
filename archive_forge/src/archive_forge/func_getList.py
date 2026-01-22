import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def getList(self, string):
    if re.search('(?<!\\\\)\\;', string):
        list = re.split('(?<!\\\\);', string)
    elif re.search('(?<!\\\\)\\|', string):
        list = re.split('(?<!\\\\)\\|', string)
    elif re.search('(?<!\\\\),', string):
        list = re.split('(?<!\\\\),', string)
    else:
        list = [string]
    if list[-1] == '':
        list.pop()
    return list