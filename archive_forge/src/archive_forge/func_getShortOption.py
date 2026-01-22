import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
def getShortOption(self, longname):
    """
        Return the short option letter or None
        @return: C{str} or L{None}
        """
    optList = self.allOptionsNameToDefinition[longname]
    return optList[0] or None