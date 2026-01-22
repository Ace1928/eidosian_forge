import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def check_line(self, context, state, transitions=None):
    """
        Examine one line of input for a transition match & execute its method.

        Parameters:

        - `context`: application-dependent storage.
        - `state`: a `State` object, the current state.
        - `transitions`: an optional ordered list of transition names to try,
          instead of ``state.transition_order``.

        Return the values returned by the transition method:

        - context: possibly modified from the parameter `context`;
        - next state name (`State` subclass name);
        - the result output of the transition, a list.

        When there is no match, ``state.no_match()`` is called and its return
        value is returned.
        """
    if transitions is None:
        transitions = state.transition_order
    state_correction = None
    if self.debug:
        print('\nStateMachine.check_line: state="%s", transitions=%r.' % (state.__class__.__name__, transitions), file=self._stderr)
    for name in transitions:
        pattern, method, next_state = state.transitions[name]
        match = pattern.match(self.line)
        if match:
            if self.debug:
                print('\nStateMachine.check_line: Matched transition "%s" in state "%s".' % (name, state.__class__.__name__), file=self._stderr)
            return method(match, context, next_state)
    else:
        if self.debug:
            print('\nStateMachine.check_line: No match in state "%s".' % state.__class__.__name__, file=self._stderr)
        return state.no_match(context, transitions)