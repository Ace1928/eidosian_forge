import os
import sys
import warnings
from shutil import get_terminal_size as _get_terminal_size
def freeze_term_title():
    warnings.warn('This function is deprecated, use toggle_set_term_title()')
    global ignore_termtitle
    ignore_termtitle = True