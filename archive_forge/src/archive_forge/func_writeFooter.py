import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
def writeFooter(self):
    """
        Write the last bit of code that finishes the call to _arguments
        @return: L{None}
        """
    self.file.write(b'&& return 0\n')