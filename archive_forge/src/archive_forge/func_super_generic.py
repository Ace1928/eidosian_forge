import textwrap
import os
import pkg_resources
from .command import Command, BadCommand
import fnmatch
import re
import traceback
from io import StringIO
import inspect
import types
def super_generic(obj):
    desc = SuperGeneric(obj)
    if not desc.description:
        return None
    return desc