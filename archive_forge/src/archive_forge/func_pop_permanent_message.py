import greenlet
import time
from curtsies import events
from ..translations import _
from ..repl import Interaction
from ..curtsiesfrontend.events import RefreshRequestEvent
from ..curtsiesfrontend.manual_readline import edit_keys
def pop_permanent_message(self, msg):
    if msg in self.permanent_stack:
        self.permanent_stack.remove(msg)
    else:
        raise ValueError('Message %r was not in permanent_stack' % msg)