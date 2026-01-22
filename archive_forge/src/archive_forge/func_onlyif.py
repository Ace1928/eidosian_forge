import os
import shutil
import sys
import tempfile
import unittest
from importlib import import_module
from decorator import decorator
from .ipunittest import ipdoctest, ipdocstring
def onlyif(condition, msg):
    """The reverse from skipif, see skipif for details."""
    return skipif(not condition, msg)