import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time
import threading
def localtrace_trace_and_count(self, frame, why, arg):
    if why == 'line':
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        key = (filename, lineno)
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.start_time:
            print('%.2f' % (_time() - self.start_time), end=' ')
        bname = os.path.basename(filename)
        print('%s(%d): %s' % (bname, lineno, linecache.getline(filename, lineno)), end='')
    return self.localtrace