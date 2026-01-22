import os
import sys
import threading
from . import process
from . import reduction
def get_all_start_methods(self):
    if sys.platform == 'win32':
        return ['spawn']
    else:
        methods = ['spawn', 'fork'] if sys.platform == 'darwin' else ['fork', 'spawn']
        if reduction.HAVE_SEND_HANDLE:
            methods.append('forkserver')
        return methods