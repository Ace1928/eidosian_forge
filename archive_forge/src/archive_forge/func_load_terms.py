import html
import json
import pickle
import re
import warnings
from importlib import import_module
from os import path
from typing import IO, Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes, package_dir
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import BuildEnvironment
from sphinx.util import split_into
from sphinx.search.en import SearchEnglish
def load_terms(mapping: Dict[str, Any]) -> Dict[str, Set[str]]:
    rv = {}
    for k, v in mapping.items():
        if isinstance(v, int):
            rv[k] = {index2fn[v]}
        else:
            rv[k] = {index2fn[i] for i in v}
    return rv