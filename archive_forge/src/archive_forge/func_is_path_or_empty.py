import argparse
import locale
import os
import sys
import time
from collections import OrderedDict
from os import path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from docutils.utils import column_width
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.locale import __
from sphinx.util.console import bold, color_terminal, colorize, nocolor, red  # type: ignore
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxRenderer
def is_path_or_empty(x: str) -> str:
    if x == '':
        return x
    return is_path(x)