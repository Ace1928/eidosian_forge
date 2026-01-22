import greenlet
import time
from curtsies import events
from ..translations import _
from ..repl import Interaction
from ..curtsiesfrontend.events import RefreshRequestEvent
from ..curtsiesfrontend.manual_readline import edit_keys
@property
def has_focus(self):
    return self.in_prompt or self.in_confirm or self.waiting_for_refresh