from . import screen
from . import FSM
import string
def do_sgr(self, fsm):
    """Select Graphic Rendition, e.g. color. """
    screen = fsm.memory[0]
    fsm.memory = [screen]