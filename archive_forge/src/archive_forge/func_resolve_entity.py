import collections
import difflib
import io
import os
import tokenize
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.util import tf_inspect
def resolve_entity(node, source, entity):
    """Like resolve, but extracts the context information from an entity."""
    lines, lineno = tf_inspect.getsourcelines(entity)
    filepath = tf_inspect.getsourcefile(entity)
    definition_line = lines[0]
    col_offset = len(definition_line) - len(definition_line.lstrip())
    resolve(node, source, filepath, lineno, col_offset)