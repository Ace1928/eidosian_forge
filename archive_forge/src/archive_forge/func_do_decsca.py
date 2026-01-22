from . import screen
from . import FSM
import string
def do_decsca(self, fsm):
    """Select character protection attribute. """
    screen = fsm.memory[0]
    fsm.memory = [screen]