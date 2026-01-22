import io
import sys
import time
import marshal
def simulate_cmd_complete(self):
    get_time = self.get_time
    t = get_time() - self.t
    while self.cur[-1]:
        self.dispatch['return'](self, self.cur[-2], t)
        t = 0
    self.t = get_time() - t