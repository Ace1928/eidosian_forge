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
def find_binary_iter(name, path_to_bin=None, env_vars=(), searchpath=(), binary_names=None, url=None, verbose=False):
    """
    Search for a file to be used by nltk.

    :param name: The name or path of the file.
    :param path_to_bin: The user-supplied binary location (deprecated)
    :param env_vars: A list of environment variable names to check.
    :param file_names: A list of alternative file names to check.
    :param searchpath: List of directories to search.
    :param url: URL presented to user for download help.
    :param verbose: Whether or not to print path when a file is found.
    """
    yield from find_file_iter(path_to_bin or name, env_vars, searchpath, binary_names, url, verbose)