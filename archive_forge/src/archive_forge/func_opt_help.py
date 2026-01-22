from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
def opt_help(self):
    """
        Display this help and exit.
        """
    print(self.__str__())
    sys.exit(0)