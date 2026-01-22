import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def lexspan(self, n):
    startpos = getattr(self.slice[n], 'lexpos', 0)
    endpos = getattr(self.slice[n], 'endlexpos', startpos)
    return (startpos, endpos)