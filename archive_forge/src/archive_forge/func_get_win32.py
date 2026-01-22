import collections
import os
import re
import sys
import functools
import itertools
def get_win32():
    return os.environ.get('PROCESSOR_IDENTIFIER', _get_machine_win32())