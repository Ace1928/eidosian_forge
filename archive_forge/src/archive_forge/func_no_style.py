import functools
import os
import sys
from django.utils import termcolors
@functools.cache
def no_style():
    """
    Return a Style object with no color scheme.
    """
    return make_style('nocolor')