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
def nested_list_parse(self, block, input_offset, node, initial_state, blank_finish, blank_finish_state=None, extra_settings={}, match_titles=False, state_machine_class=None, state_machine_kwargs=None):
    """
        Create a new StateMachine rooted at `node` and run it over the input
        `block`. Also keep track of optional intermediate blank lines and the
        required final one.
        """
    if state_machine_class is None:
        state_machine_class = self.nested_sm
    if state_machine_kwargs is None:
        state_machine_kwargs = self.nested_sm_kwargs.copy()
    state_machine_kwargs['initial_state'] = initial_state
    state_machine = state_machine_class(debug=self.debug, **state_machine_kwargs)
    if blank_finish_state is None:
        blank_finish_state = initial_state
    state_machine.states[blank_finish_state].blank_finish = blank_finish
    for key, value in list(extra_settings.items()):
        setattr(state_machine.states[initial_state], key, value)
    state_machine.run(block, input_offset, memo=self.memo, node=node, match_titles=match_titles)
    blank_finish = state_machine.states[blank_finish_state].blank_finish
    state_machine.unlink()
    return (state_machine.abs_line_offset(), blank_finish)