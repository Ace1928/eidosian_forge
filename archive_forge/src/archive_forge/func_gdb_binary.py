from collections import namedtuple
import os
import re
import subprocess
from textwrap import dedent
from numba import config
@property
def gdb_binary(self):
    return self._gdb_binary