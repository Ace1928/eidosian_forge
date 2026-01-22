import os, sys
from OpenGL.plugins import PlatformPlugin
from OpenGL import _configflags
def createFunction(function, dll, extension, deprecated=False, error_checker=None, force_extension=False):
    """Allows the more compact declaration format to use the old-style constructor"""
    return nullFunction(function.__name__, dll or PLATFORM.GL, resultType=function.resultType, argTypes=function.argTypes, doc=None, argNames=function.argNames, extension=extension, deprecated=deprecated, module=function.__module__, error_checker=error_checker, force_extension=force_extension or getattr(function, 'force_extension', force_extension))