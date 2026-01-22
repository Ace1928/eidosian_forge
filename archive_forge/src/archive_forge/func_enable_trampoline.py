from collections import deque
from threading import local
def enable_trampoline(self):
    self.trampoline_enabled = True