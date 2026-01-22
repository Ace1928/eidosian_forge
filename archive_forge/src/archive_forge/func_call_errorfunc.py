import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def call_errorfunc(errorfunc, token, parser):
    global _errok, _token, _restart
    _errok = parser.errok
    _token = parser.token
    _restart = parser.restart
    r = errorfunc(token)
    try:
        del _errok, _token, _restart
    except NameError:
        pass
    return r