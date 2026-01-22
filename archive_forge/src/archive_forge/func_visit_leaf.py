import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def visit_leaf(self, leaf):
    if leaf.type == 'error_leaf':
        if leaf.token_type in ('INDENT', 'ERROR_DEDENT'):
            spacing = list(leaf.get_next_leaf()._split_prefix())[-1]
            if leaf.token_type == 'INDENT':
                message = 'unexpected indent'
            else:
                message = 'unindent does not match any outer indentation level'
            self._add_indentation_error(spacing, message)
        else:
            if leaf.value.startswith('\\'):
                message = 'unexpected character after line continuation character'
            else:
                match = re.match('\\w{,2}("{1,3}|\'{1,3})', leaf.value)
                if match is None:
                    message = 'invalid syntax'
                    if self.version >= (3, 9) and leaf.value in _get_token_collection(self.version).always_break_tokens:
                        message = 'f-string: ' + message
                elif len(match.group(1)) == 1:
                    message = 'EOL while scanning string literal'
                else:
                    message = 'EOF while scanning triple-quoted string literal'
            self._add_syntax_error(leaf, message)
        return ''
    elif leaf.value == ':':
        parent = leaf.parent
        if parent.type in ('classdef', 'funcdef'):
            self.context = self.context.add_context(parent)
    return super().visit_leaf(leaf)