import os
import subprocess
import sys
from .error import TryNext
def show_in_pager(self, data, start, screen_lines):
    """ Run a string through pager """
    raise TryNext