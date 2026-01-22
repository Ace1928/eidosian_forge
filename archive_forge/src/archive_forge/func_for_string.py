import re
import tokenize
from collections import OrderedDict
from importlib import import_module
from inspect import Signature
from os import path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile
from sphinx.errors import PycodeError
from sphinx.pycode.parser import Parser
@classmethod
def for_string(cls, string: str, modname: str, srcname: str='<string>') -> 'ModuleAnalyzer':
    return cls(string, modname, srcname)