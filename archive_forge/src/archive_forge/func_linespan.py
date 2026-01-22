import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def linespan(self, n):
    startline = getattr(self.slice[n], 'lineno', 0)
    endline = getattr(self.slice[n], 'endlineno', startline)
    return (startline, endline)