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
def valid_dir(d: Dict) -> bool:
    dir = d['path']
    if not path.exists(dir):
        return True
    if not path.isdir(dir):
        return False
    if {'Makefile', 'make.bat'} & set(os.listdir(dir)):
        return False
    if d['sep']:
        dir = os.path.join('source', dir)
        if not path.exists(dir):
            return True
        if not path.isdir(dir):
            return False
    reserved_names = ['conf.py', d['dot'] + 'static', d['dot'] + 'templates', d['master'] + d['suffix']]
    if set(reserved_names) & set(os.listdir(dir)):
        return False
    return True