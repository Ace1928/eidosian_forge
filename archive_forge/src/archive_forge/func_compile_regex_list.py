import glob
import inspect
import pickle
import re
from importlib import import_module
from os import path
from typing import IO, Any, Dict, List, Pattern, Set, Tuple
import sphinx
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import red  # type: ignore
from sphinx.util.inspect import safe_getattr
def compile_regex_list(name: str, exps: str) -> List[Pattern[str]]:
    lst = []
    for exp in exps:
        try:
            lst.append(re.compile(exp))
        except Exception:
            logger.warning(__('invalid regex %r in %s'), exp, name)
    return lst