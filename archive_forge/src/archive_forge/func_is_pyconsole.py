import sys
from typing import Any, Dict, List, NamedTuple
from docutils import nodes
from docutils.nodes import Node, TextElement
from pygments.lexers import PythonConsoleLexer, guess_lexer
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.ext import doctest
from sphinx.transforms import SphinxTransform
@staticmethod
def is_pyconsole(node: nodes.literal_block) -> bool:
    if node.rawsource != node.astext():
        return False
    language = node.get('language')
    if language in {'pycon', 'pycon3'}:
        return True
    elif language in {'py', 'python', 'py3', 'python3', 'default'}:
        return node.rawsource.startswith('>>>')
    elif language == 'guess':
        try:
            lexer = guess_lexer(node.rawsource)
            return isinstance(lexer, PythonConsoleLexer)
        except Exception:
            pass
    return False