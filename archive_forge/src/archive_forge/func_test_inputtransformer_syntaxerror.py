import sys
import unittest
import os
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from IPython.testing import tools as tt
from IPython.terminal.ptutils import _elide, _adjust_completion_text_based_on_context
from IPython.terminal.shortcuts.auto_suggest import NavigableAutoSuggestFromHistory
@mock_input
def test_inputtransformer_syntaxerror(self):
    ip = get_ipython()
    ip.input_transformers_post.append(syntax_error_transformer)
    try:
        with tt.AssertPrints('4', suppress=False):
            yield u'print(2*2)'
        with tt.AssertPrints('SyntaxError: input contains', suppress=False):
            yield u'print(2345) # syntaxerror'
        with tt.AssertPrints('16', suppress=False):
            yield u'print(4*4)'
    finally:
        ip.input_transformers_post.remove(syntax_error_transformer)