from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
def getUsage(self, width: Optional[int]=None) -> str:
    if hasattr(self, 'subOptions'):
        return cast(Options, self.subOptions).getUsage(width=width)
    if not width:
        width = int(os.environ.get('COLUMNS', '80'))
    if hasattr(self, 'subCommands'):
        cmdDicts = []
        for cmd, short, parser, desc in self.subCommands:
            cmdDicts.append({'long': cmd, 'short': short, 'doc': desc, 'optType': 'command', 'default': None})
        chunks = docMakeChunks(cmdDicts, width)
        commands = 'Commands:\n' + ''.join(chunks)
    else:
        commands = ''
    longToShort = {}
    for key, value in self.synonyms.items():
        longname = value
        if key != longname and len(key) == 1:
            longToShort[longname] = key
        elif longname not in longToShort:
            longToShort[longname] = None
        else:
            pass
    optDicts = []
    for opt in self.longOpt:
        if opt[-1] == '=':
            optType = 'parameter'
            opt = opt[:-1]
        else:
            optType = 'flag'
        optDicts.append({'long': opt, 'short': longToShort[opt], 'doc': self.docs[opt], 'optType': optType, 'default': self.defaults.get(opt, None), 'dispatch': self._dispatch.get(opt, None)})
    if not getattr(self, 'longdesc', None) is None:
        longdesc = cast(str, self.longdesc)
    else:
        import __main__
        if getattr(__main__, '__doc__', None):
            longdesc = __main__.__doc__
        else:
            longdesc = ''
    if longdesc:
        longdesc = '\n' + '\n'.join(textwrap.wrap(longdesc, width)).strip() + '\n'
    if optDicts:
        chunks = docMakeChunks(optDicts, width)
        s = 'Options:\n%s' % ''.join(chunks)
    else:
        s = 'Options: None\n'
    return s + longdesc + commands