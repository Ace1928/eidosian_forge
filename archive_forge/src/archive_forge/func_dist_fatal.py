import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
@staticmethod
def dist_fatal(*args):
    """Raise a distutils error"""
    from distutils.errors import DistutilsError
    raise DistutilsError(_Distutils._dist_str(*args))