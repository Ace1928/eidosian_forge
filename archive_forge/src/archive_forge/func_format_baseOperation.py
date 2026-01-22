import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
def format_baseOperation(self, property, value):
    """Format a baseOperation reference for display"""
    if hasattr(value, '__name__'):
        return '%s = %s' % (property, value.__name__)
    else:
        return '%s = %r' % (property, value)