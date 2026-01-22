import io
import sys
import time
import marshal
def trace_dispatch_mac(self, frame, event, arg):
    timer = self.timer
    t = timer() / 60.0 - self.t - self.bias
    if event == 'c_call':
        self.c_func_name = arg.__name__
    if self.dispatch[event](self, frame, t):
        self.t = timer() / 60.0
    else:
        self.t = timer() / 60.0 - t