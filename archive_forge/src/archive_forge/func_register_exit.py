import sys
import warnings
from collections import deque
from functools import wraps
def register_exit(self, callback):
    return self.push(callback)