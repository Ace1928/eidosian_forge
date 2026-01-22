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
def word_filter(self, word: str) -> bool:
    """
        Return true if the target word should be registered in the search index.
        This method is called after stemming.
        """
    return len(word) == 0 or not (len(word) < 3 and 12353 < ord(word[0]) < 12436 or (ord(word[0]) < 256 and word in self.stopwords))