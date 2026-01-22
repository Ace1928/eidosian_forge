import io
import sys
import time
import marshal
def trace_dispatch_l(self, frame, event, arg):
    get_time = self.get_time
    t = get_time() - self.t - self.bias
    if event == 'c_call':
        self.c_func_name = arg.__name__
    if self.dispatch[event](self, frame, t):
        self.t = get_time()
    else:
        self.t = get_time() - t