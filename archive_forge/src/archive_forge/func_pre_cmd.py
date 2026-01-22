import os
import re
import shlex
import sys
import pygments
from pathlib import Path
from IPython.utils.text import marquee
from IPython.utils import openpy
from IPython.utils import py3compat
def pre_cmd(self):
    """Method called before executing each block.

        This one simply clears the screen."""
    from IPython.utils.terminal import _term_clear
    _term_clear()