from collections import namedtuple
import os
import re
import subprocess
from textwrap import dedent
from numba import config
def check_launch(self):
    """Checks that gdb will launch ok"""
    return self._run_cmd()