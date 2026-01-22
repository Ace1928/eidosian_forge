import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def search_namedexpr_in_comp_for(node):
    while True:
        parent = node.parent
        if parent is None:
            return parent
        if parent.type == 'sync_comp_for' and parent.children[3] == node:
            return parent
        node = parent