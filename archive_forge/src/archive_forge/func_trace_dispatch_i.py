import io
import sys
import time
import marshal
def trace_dispatch_i(self, frame, event, arg):
    timer = self.timer
    t = timer() - self.t - self.bias
    if event == 'c_call':
        self.c_func_name = arg.__name__
    if self.dispatch[event](self, frame, t):
        self.t = timer()
    else:
        self.t = timer() - t