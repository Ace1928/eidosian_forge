import errno
import sys
from types import FunctionType
import eventlet
from eventlet import greenio
from eventlet import patcher
from eventlet.green import select, threading, time
from eventlet.green import selectors
def patched_function(function):
    new_function = FunctionType(function.__code__, globals())
    new_function.__kwdefaults__ = function.__kwdefaults__
    new_function.__defaults__ = function.__defaults__
    return new_function