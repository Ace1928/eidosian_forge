import sys
import re
from types import FunctionType, MethodType
from docutils import nodes, statemachine, utils
from docutils import ApplicationError, DataError
from docutils.statemachine import StateMachineWS, StateWS
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.nodes import whitespace_normalize_name
import docutils.parsers.rst
from docutils.parsers.rst import directives, languages, tableparser, roles
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils import escape2null, unescape, column_width
from docutils.utils import punctuation_chars, roman, urischemes
from docutils.utils import split_escaped_whitespace
def nested_parse(self, block, input_offset, node, match_titles=False, state_machine_class=None, state_machine_kwargs=None):
    """
        Create a new StateMachine rooted at `node` and run it over the input
        `block`.
        """
    use_default = 0
    if state_machine_class is None:
        state_machine_class = self.nested_sm
        use_default += 1
    if state_machine_kwargs is None:
        state_machine_kwargs = self.nested_sm_kwargs
        use_default += 1
    block_length = len(block)
    state_machine = None
    if use_default == 2:
        try:
            state_machine = self.nested_sm_cache.pop()
        except IndexError:
            pass
    if not state_machine:
        state_machine = state_machine_class(debug=self.debug, **state_machine_kwargs)
    state_machine.run(block, input_offset, memo=self.memo, node=node, match_titles=match_titles)
    if use_default == 2:
        self.nested_sm_cache.append(state_machine)
    else:
        state_machine.unlink()
    new_offset = state_machine.abs_line_offset()
    if block.parent and len(block) - block_length != 0:
        self.state_machine.next_line(len(block) - block_length)
    return new_offset