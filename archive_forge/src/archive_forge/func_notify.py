import greenlet
import time
from curtsies import events
from ..translations import _
from ..repl import Interaction
from ..curtsiesfrontend.events import RefreshRequestEvent
from ..curtsiesfrontend.manual_readline import edit_keys
def notify(self, msg, n=3.0, wait_for_keypress=False):
    self.request_context = greenlet.getcurrent()
    self.message_time = n
    self.message(msg, schedule_refresh=wait_for_keypress)
    self.waiting_for_refresh = True
    self.request_refresh()
    self.main_context.switch(msg)