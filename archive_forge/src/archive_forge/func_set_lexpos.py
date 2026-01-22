import re
import types
import sys
import os.path
import inspect
import warnings
def set_lexpos(self, n, lexpos):
    self.slice[n].lexpos = lexpos