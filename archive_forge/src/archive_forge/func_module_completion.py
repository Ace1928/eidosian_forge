import glob
import inspect
import os
import re
import sys
from importlib import import_module
from importlib.machinery import all_suffixes
from time import time
from zipimport import zipimporter
from .completer import expand_user, compress_user
from .error import TryNext
from ..utils._process_common import arg_split
from IPython import get_ipython
from typing import List
import_re = re.compile(r'(?P<name>[^\W\d]\w*?)'
def module_completion(line):
    """
    Returns a list containing the completion possibilities for an import line.

    The line looks like this :
    'import xml.d'
    'from xml.dom import'
    """
    words = line.split(' ')
    nwords = len(words)
    if nwords == 3 and words[0] == 'from':
        return ['import ']
    if nwords < 3 and words[0] in {'%aimport', 'import', 'from'}:
        if nwords == 1:
            return get_root_modules()
        mod = words[1].split('.')
        if len(mod) < 2:
            return get_root_modules()
        completion_list = try_import('.'.join(mod[:-1]), True)
        return ['.'.join(mod[:-1] + [el]) for el in completion_list]
    if nwords >= 3 and words[0] == 'from':
        mod = words[1]
        return try_import(mod)