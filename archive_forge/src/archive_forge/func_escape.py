import greenlet
import time
from curtsies import events
from ..translations import _
from ..repl import Interaction
from ..curtsiesfrontend.events import RefreshRequestEvent
from ..curtsiesfrontend.manual_readline import edit_keys
def escape(self):
    """unfocus from statusbar, clear prompt state, wait for notify call"""
    self.in_prompt = False
    self.in_confirm = False
    self.prompt = ''
    self._current_line = ''