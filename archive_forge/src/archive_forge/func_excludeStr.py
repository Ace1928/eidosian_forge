import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
def excludeStr(self, longname, buildShort=False):
    """
        Generate an "exclusion string" for the given option

        @type longname: C{str}
        @param longname: The long option name (e.g. "verbose" instead of "v")

        @type buildShort: C{bool}
        @param buildShort: May be True to indicate we're building an excludes
            string for the short option that corresponds to the given long opt.

        @return: The generated C{str}
        """
    if longname in self.excludes:
        exclusions = self.excludes[longname].copy()
    else:
        exclusions = set()
    if longname not in self.multiUse:
        if buildShort is False:
            short = self.getShortOption(longname)
            if short is not None:
                exclusions.add(short)
        else:
            exclusions.add(longname)
    if not exclusions:
        return ''
    strings = []
    for optName in exclusions:
        if len(optName) == 1:
            strings.append('-' + optName)
        else:
            strings.append('--' + optName)
    strings.sort()
    return '(%s)' % ' '.join(strings)