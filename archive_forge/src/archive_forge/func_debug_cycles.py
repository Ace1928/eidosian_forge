import gc
from ..Qt import QtCore
def debug_cycles(self):
    gc.collect()
    for obj in gc.garbage:
        print(obj, repr(obj), type(obj))