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
def context_for_searchtool(self) -> Dict[str, Any]:
    if self.lang.js_splitter_code:
        js_splitter_code = self.lang.js_splitter_code
    else:
        js_splitter_code = self.js_splitter_code
    return {'search_language_stemming_code': self.get_js_stemmer_code(), 'search_language_stop_words': json.dumps(sorted(self.lang.stopwords)), 'search_scorer_tool': self.js_scorer_code, 'search_word_splitter_code': js_splitter_code}