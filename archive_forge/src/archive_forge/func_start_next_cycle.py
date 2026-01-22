from typing import cast, Dict, Optional, Set, Tuple, Type, Union
from ._events import *
from ._util import LocalProtocolError, Sentinel
def start_next_cycle(self) -> None:
    if self.states != {CLIENT: DONE, SERVER: DONE}:
        raise LocalProtocolError('not in a reusable state. self.states={}'.format(self.states))
    assert self.keep_alive
    assert not self.pending_switch_proposals
    self.states = {CLIENT: IDLE, SERVER: IDLE}