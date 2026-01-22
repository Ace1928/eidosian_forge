import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def get_style_defs(self, arg=''):
    raise NotImplementedError('The -S option is meaningless for the image formatter. Use -O style=<stylename> instead.')