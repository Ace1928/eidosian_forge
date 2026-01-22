import fnmatch
import locale
import os
import re
import stat
import subprocess
import sys
import textwrap
import types
import warnings
from xml.etree import ElementTree
def find_binary(name, path_to_bin=None, env_vars=(), searchpath=(), binary_names=None, url=None, verbose=False):
    return next(find_binary_iter(name, path_to_bin, env_vars, searchpath, binary_names, url, verbose))