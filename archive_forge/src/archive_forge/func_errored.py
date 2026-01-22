import time
from .exceptions import EOF, TIMEOUT
def errored(self):
    spawn = self.spawn
    spawn.before = spawn._before.getvalue()
    spawn.after = None
    spawn.match = None
    spawn.match_index = None