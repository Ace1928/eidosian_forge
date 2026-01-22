import warnings
from typing import TYPE_CHECKING, List, NewType
from outlines.fsm.guide import CFGGuide, RegexGuide, StopAtEOSGuide
def next_state(self, state: FSMState, token_id: int) -> FSMState:
    return FSMState(self.get_next_state(state, token_id))